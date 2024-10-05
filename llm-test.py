from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse, Response

import ray
from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath, BaseModelPath
from vllm.utils import FlexibleArgumentParser

import huggingface_hub

from vllm import LLM, SamplingParams

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_ongoing_requests": 5,
    },
    max_ongoing_requests=10,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None,
    ):

        # Ray actors for different parts of the model
        @ray.remote(num_gpus=1, resources={"accelerator_type_960": 1})
        class SmallLayerWorker:
            def __init__(self, engine_args):
                # Load only a subset of layers that fit into 2GB VRAM
                self.engine = AsyncLLMEngine.from_engine_args(engine_args)
                self.model_part = self.engine.model.transformer.h[:6].cuda()  # First 6 layers

            async def forward(self, inputs):
                """Process the input through the first part of the model."""
                with torch.no_grad():
                    inputs = inputs.to("cuda")
                    outputs = self.model_part(inputs)
                    return outputs.cpu()  # Send outputs to the next worker

        @ray.remote(num_gpus=1, resources={"accelerator_type_3070": 1})
        class LargeLayerWorker:
            def __init__(self, engine_args):
                # Load the second part of the model that fits into 8GB VRAM
                self.engine = AsyncLLMEngine.from_engine_args(engine_args)
                self.model_part = self.engine.model.transformer.h[6:].cuda()  # Remaining layers

            async def forward(self, inputs):
                """Process the input through the second part of the model."""
                with torch.no_grad():
                    inputs = inputs.to("cuda")
                    outputs = self.model_part(inputs)
                    return outputs.cpu()  # Final output


        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)        

        # Create Ray actors for each pipeline stage (small VRAM vs. large VRAM)
        self.small_vram_worker = SmallLayerWorker.remote(engine_args)  # For the node with 2GB VRAM
        self.large_vram_worker = LargeLayerWorker.remote(engine_args)  # For the node with 8GB VRAM

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            # Determine the name of the served model for the OpenAI client.
            # if self.engine_args.served_model_name is not None:
            #     served_model_names = self.engine_args.served_model_name
            # else:
            #     served_model_names = [self.engine_args.model]
            base_model_paths = [BaseModelPath(name="TinyLlama-1.1B-Chat-v1.0", model_path="/tmp/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf")]
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                base_model_paths,
                self.response_role,
                lora_modules=self.lora_modules,
                chat_template=self.chat_template,
                prompt_adapters=None,
                request_logger=None,
            )
        logger.info(f"Request: {request}")
        # generator = await self.openai_serving_chat.create_chat_completion(
        #     request, raw_request
        # )

        # Execute the first part of the model on the small VRAM worker (2GB node)
        intermediate_output = await self.small_vram_worker.forward.remote(request)

        # Execute the second part on the large VRAM worker (8GB node)
        final_output = await self.large_vram_worker.forward.remote(intermediate_output)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)

    filename = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
    local_dir = "/tmp/models/"
    model = f'{local_dir}{filename}'
    # model = "/tmp/models/TinyLlama-1.1B-Chat-v1.0"

    parsed_args.model = model
    parsed_args.tensor_parallel_size = 1
    parsed_args.pipeline_parallel_size = 2
    parsed_args.gpu_memory_utilization = 0.8

    parsed_args.tokenizer="/tmp/models/TinyLlama-1.1B-Chat-v1.0"

    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    tp = engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    # pg_resources = []
    # pg_resources.append({"CPU": 1})  # for the deployment replica
    # for i in range(tp):
    #     pg_resources.append({"CPU": 1, "GPU": 1})  # for the vLLM actors

    pg_resources = [
        {"CPU": 1, "GPU": 1},  # For the small VRAM worker (2GB)
        {"CPU": 1, "GPU": 1},  # For the large VRAM worker (8GB)
    ]

    # We use the "STRICT_SPREAD" strategy below to ensure all vLLM actors are placed on
    # different Ray nodes.
    return VLLMDeployment.options(
        placement_group_bundles=pg_resources, placement_group_strategy="STRICT_SPREAD"
    ).bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.chat_template,
    )