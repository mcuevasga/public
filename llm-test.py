from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse, Response

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
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)        

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
            base_model_paths = [BaseModelPath(name="Llama-3.2-3B-Instruct", model_path=self.engine_args.model)]
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
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )

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

    model_file = "llama2_7b_chat_uncensored.Q8_0.gguf"
    model_folder = "Llama-3.2-3B-Instruct"
    local_dir = "/data/models/cache/"
    model_model_file = f'{local_dir}{model_file}'
    model_model_folder = f'{local_dir}{model_folder}'

    parsed_args.model = model_model_folder
    parsed_args.tensor_parallel_size = 1
    parsed_args.pipeline_parallel_size = 2
    parsed_args.gpu_memory_utilization = 0.5
    parsed_args.max_num_seqs = 1
    parsed_args.max_model_len = 45000

    # template_str =chat_template = "<s>[INST] <<SYS>>\n{your_system_message}\n<</SYS>>\n\n{user_message_1} [/INST]"
    # parsed_args.chat_template="/data/models/cache/llama2_7b_chat_uncensored/template.jinja"

    # parsed_args.chat_template=template_str

    parsed_args.tokenizer=model_model_folder

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

    tp = engine_args.tensor_parallel_size * parsed_args.pipeline_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    # pg_resources.append({"CPU": 1})  # for the deployment replica
    for i in range(tp):
        pg_resources.append({"CPU": 1, "GPU": 1})  # for the vLLM actors

    # pg_resources = [
    #     {"CPU": 1, "GPU": 1},  # For the small VRAM worker (2GB)
    #     {"CPU": 1, "GPU": 1},  # For the large VRAM worker (8GB)
    # ]

    # We use the "STRICT_SPREAD" strategy below to ensure all vLLM actors are placed on
    # different Ray nodes.
    return VLLMDeployment.options(
        placement_group_bundles=pg_resources, placement_group_strategy="SPREAD"
    ).bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.chat_template,
    )
