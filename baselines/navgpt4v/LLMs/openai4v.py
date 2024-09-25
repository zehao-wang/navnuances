from __future__ import annotations

import logging
import sys
import warnings
from typing import (
    AbstractSet,
    Any,
    AsyncIterator,
    Callable,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)
from langchain.llms.openai import OpenAI, BaseOpenAI
from pydantic import Field, root_validator
import os
from PIL import Image
import base64
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import BaseLLM, create_base_retry_decorator
from langchain.schema import Generation, LLMResult
from langchain.schema.output import GenerationChunk
from langchain.utils import get_from_dict_or_env, get_pydantic_field_names
import time
import io
logger = logging.getLogger(__name__)

def construct_img_request(bs64_datum):
    return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{bs64_datum}",
                "detail": "low"
            }
        }

def construct_text_request(content):
    return {
            "type": "text",
            "text": content
        }

def update_token_usage(
    keys: Set[str], response: Dict[str, Any], token_usage: Dict[str, Any]
) -> None:
    """Update token usage."""
    _keys_to_use = keys.intersection(response["usage"])
    for _key in _keys_to_use:
        if _key not in token_usage:
            token_usage[_key] = response["usage"][_key]
        else:
            token_usage[_key] += response["usage"][_key]


def _create_retry_decorator(
    llm: Union[BaseOpenAI, OpenAIChat4v],
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    import openai

    errors = [
        openai.error.Timeout,
        openai.error.APIError,
        openai.error.APIConnectionError,
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )


def completion_with_retry(
    llm: Union[BaseOpenAI, OpenAIChat4v],
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        return llm.client.create(**kwargs)

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(
    llm: Union[BaseOpenAI, OpenAIChat4v],
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        # Use OpenAI's async api https://github.com/openai/openai-python#async-api
        return await llm.client.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)


class OpenAIChat4v(BaseLLM):
    """OpenAI Chat large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.llms import OpenAIChat
            openaichat = OpenAIChat(model_name="gpt-3.5-turbo")
    """

    client: Any  #: :meta private:
    model_name: str = "gpt-3.5-turbo"
    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    max_retries: int = 10
    """Maximum number of retries to make when generating."""
    prefix_messages: List = Field(default_factory=list)
    """Series of messages for Chat input."""
    streaming: bool = False
    """Whether to stream the results or not."""
    allowed_special: Union[Literal["all"], AbstractSet[str]] = set()
    """Set of special tokens that are allowed。"""
    disallowed_special: Union[Literal["all"], Collection[str]] = "all"
    """Set of special tokens that are not allowed。"""

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        openai_api_base = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="",
        )
        openai_proxy = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        openai_organization = get_from_dict_or_env(
            values, "openai_organization", "OPENAI_ORGANIZATION", default=""
        )
        try:
            import openai

            openai.api_key = openai_api_key
            if openai_api_base:
                openai.api_base = openai_api_base
            if openai_organization:
                openai.organization = openai_organization
            if openai_proxy:
                openai.proxy = {"http": openai_proxy, "https": openai_proxy}  # type: ignore[assignment]  # noqa: E501
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        try:
            values["client"] = openai.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        warnings.warn(
            "You are trying to use a chat model. This way of initializing it is "
            "no longer supported. Instead, please use: "
            "`from langchain.chat_models import ChatOpenAI`"
        )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return self.model_kwargs

    def generate_prompt(
        self,
        prompts,
        stop: Optional[List[str]] = None,
        callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_strings = [p for p in prompts]
        return self.generate(prompt_strings, stop=stop, callbacks=callbacks, **kwargs)
    
    def _get_chat_params(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> Tuple:
        # if len(prompts) > 1:
        #     raise ValueError(
        #         f"OpenAIChat currently only supports single prompt, got {prompts}"
        #     )
        curr_msg = {"role": "user", "content": []}
        for prompt in prompts:
            if prompt[0] == 'text':
                curr_msg['content'].append(construct_text_request(prompt[1]))
            elif prompt[0] == 'vision':
                curr_msg['content'].append(construct_img_request(prompt[1]))
            else:
                import ipdb;ipdb.set_trace() # breakpoint 277
        
        messages = self.prefix_messages + [curr_msg]

        # messages = self.prefix_messages + [{"role": "user", "content": prompts[0]}]
        params: Dict[str, Any] = {**{"model": self.model_name}, **self._default_params}
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        if params.get("max_tokens") == -1:
            # for ChatGPT api, omitting max_tokens is equivalent to having no limit
            del params["max_tokens"]
        return messages, params

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        import ipdb;ipdb.set_trace() # breakpoint 266
        messages, params = self._get_chat_params([prompt], stop)
        params = {**params, **kwargs, "stream": True}
        for stream_resp in completion_with_retry(
            self, messages=messages, run_manager=run_manager, **params
        ):
            token = stream_resp["choices"][0]["delta"].get("content", "")
            yield GenerationChunk(text=token)
            if run_manager:
                run_manager.on_llm_new_token(token)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        messages, params = self._get_chat_params([prompt], stop)
        params = {**params, **kwargs, "stream": True}
        async for stream_resp in await acompletion_with_retry(
            self, messages=messages, run_manager=run_manager, **params
        ):
            token = stream_resp["choices"][0]["delta"].get("content", "")
            yield GenerationChunk(text=token)
            if run_manager:
                await run_manager.on_llm_new_token(token)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        if self.streaming:
            generation: Optional[GenerationChunk] = None
            for chunk in self._stream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return LLMResult(generations=[[generation]])

        messages, params = self._get_chat_params(prompts, stop)
        params = {**params, **kwargs}

        while True:
            try:
                full_response = completion_with_retry(
                    self, messages=messages, run_manager=run_manager, **params
                )
                text=full_response["choices"][0]["message"]["content"]
                if "invalid" in text.lower():
                    import ipdb;ipdb.set_trace() # breakpoint 355
                    raise ValueError()
                break
            except Exception as e:
                print(e)
                import ipdb;ipdb.set_trace() # breakpoint 329
                continue
            

        llm_output = {
            "token_usage": full_response["usage"],
            "model_name": self.model_name,
        } 

        if os.environ.get('DEBUG', False):
            os.makedirs('../data/R2R/exprs/log_vis', exist_ok=True )
            time_stamp = int(time.time())
            for item in messages[0]['content']:
                if 'image_url' in item.keys():
                    img_content = item['image_url']['url'].replace('data:image/jpeg;base64,', '')
                    base64_decoded = base64.b64decode(img_content)
                    image = Image.open(io.BytesIO(base64_decoded))
                    image.save(f'../data/R2R/exprs/log_vis/{time_stamp}.png')
                    break

            with open('../data/R2R/exprs/chat_history.log', 'a') as f:
                gpt_input = messages[0]['content'][0]['text']
                gpt_input = f"============== Round {time_stamp} ==============\n"+gpt_input
                gpt_input += '\n'
                gpt_input += "="*20
                gpt_input += '\n'
                gpt_input += full_response["choices"][0]["message"]["content"]
                gpt_input += "\n\n\n"
                f.writelines([gpt_input])
        # print(messages[0]['content'][0]['text'])
        # print("="*20)
        # print(full_response["choices"][0]["message"]["content"])
        # import ipdb;ipdb.set_trace() # breakpoint 354

        return LLMResult(
            generations=[
                [Generation(text=full_response["choices"][0]["message"]["content"])]
            ],
            llm_output=llm_output,
        )

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        if self.streaming:
            generation: Optional[GenerationChunk] = None
            async for chunk in self._astream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return LLMResult(generations=[[generation]])

        messages, params = self._get_chat_params(prompts, stop)
        params = {**params, **kwargs}
        full_response = await acompletion_with_retry(
            self, messages=messages, run_manager=run_manager, **params
        )
        llm_output = {
            "token_usage": full_response["usage"],
            "model_name": self.model_name,
        }
            
        return LLMResult(
            generations=[
                [Generation(text=full_response["choices"][0]["message"]["content"])]
            ],
            llm_output=llm_output,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "openai-chat"

    def get_token_ids(self, text: str) -> List[int]:
        """Get the token IDs using the tiktoken package."""
        # tiktoken NOT supported for Python < 3.8
        if sys.version_info[1] < 8:
            return super().get_token_ids(text)
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please install it with `pip install tiktoken`."
            )

        enc = tiktoken.encoding_for_model(self.model_name)
        return enc.encode(
            text,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )
