import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
)
from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig


def test_hosted_vllm_chat_transformation_file_url():
    config = HostedVLLMChatConfig()
    video_url = "https://example.com/video.mp4"
    video_data = f"data:video/mp4;base64,{video_url}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_data": video_data,
                    },
                }
            ],
        }
    ]
    transformed_response = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert transformed_response["messages"] == [
        {
            "role": "user",
            "content": [{"type": "video_url", "video_url": {"url": video_data}}],
        }
    ]


def test_hosted_vllm_chat_transformation_with_audio_url():
    from litellm import completion

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama-3.1-70b-instruct",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_response.text = json.dumps(mock_response.json.return_value)
    mock_client.post.return_value = mock_response

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
        return_value=mock_client,
    ):
        try:
            completion(
                model="hosted_vllm/llama-3.1-70b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {"url": "https://example.com/audio.mp3"},
                            },
                        ],
                    },
                ],
                api_base="https://test-vllm.example.com/v1",
            )
        except Exception:
            pass

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args[1]
        request_data = json.loads(call_kwargs["data"])
        assert request_data["messages"] == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "https://example.com/audio.mp3"},
                    }
                ],
            }
        ]


def test_hosted_vllm_supports_reasoning_effort():
    config = HostedVLLMChatConfig()
    supported_params = config.get_supported_openai_params(
        model="hosted_vllm/gpt-oss-120b"
    )
    assert "reasoning_effort" in supported_params
    optional_params = config.map_openai_params(
        non_default_params={"reasoning_effort": "high"},
        optional_params={},
        model="hosted_vllm/gpt-oss-120b",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "high"


def test_hosted_vllm_supports_thinking():
    """
    Test that hosted_vllm supports the 'thinking' parameter.

    Anthropic-style thinking is converted to OpenAI-style reasoning_effort
    since vLLM is OpenAI-compatible.

    Related issue: https://github.com/BerriAI/litellm/issues/19761
    """
    config = HostedVLLMChatConfig()
    supported_params = config.get_supported_openai_params(
        model="hosted_vllm/GLM-4.6-FP8"
    )
    assert "thinking" in supported_params

    # Test thinking below the low threshold -> "minimal"
    optional_params = config.map_openai_params(
        non_default_params={
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET - 1,
            }
        },
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert "thinking" not in optional_params  # thinking should NOT be passed
    assert optional_params["reasoning_effort"] == "minimal"

    # Test thinking with high budget_tokens -> "high"
    optional_params = config.map_openai_params(
        non_default_params={
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
            }
        },
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "high"

    # Test that existing reasoning_effort is not overwritten
    optional_params = config.map_openai_params(
        non_default_params={
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
            },
            "reasoning_effort": "low",
        },
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "low"


def _transform_single_assistant_message(assistant_message: dict) -> dict:
    return HostedVLLMChatConfig().transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=[assistant_message],
        optional_params={},
        litellm_params={},
        headers={},
    )["messages"][0]


def test_hosted_vllm_thinking_blocks_become_reasoning_fields():
    """
    thinking_blocks are not a wire format vLLM understands, so they must be flattened into
    the `reasoning` / `reasoning_content` fields instead of being dropped on the floor.
    """
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Here is my answer.",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Let me reason about this...",
                    "signature": "abc123",
                }
            ],
        },
        {
            "role": "user",
            "content": "Follow up question",
        },
    ]
    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assistant_msg = transformed["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "Here is my answer."
    assert assistant_msg["reasoning"] == "Let me reason about this..."
    assert assistant_msg["reasoning_content"] == "Let me reason about this..."
    assert "thinking_blocks" not in assistant_msg
    assert "signature" not in json.dumps(assistant_msg)


def test_hosted_vllm_multiple_thinking_blocks_are_joined_in_order():
    assistant_msg = _transform_single_assistant_message(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Response text"}],
            "thinking_blocks": [
                {"type": "thinking", "thinking": "Step 1 reasoning", "signature": "sig1"},
                {"type": "thinking", "thinking": "Step 2 reasoning", "signature": "sig2"},
            ],
        }
    )
    assert assistant_msg["content"] == "Response text"
    assert assistant_msg["reasoning"] == "Step 1 reasoning\nStep 2 reasoning"
    assert assistant_msg["reasoning_content"] == "Step 1 reasoning\nStep 2 reasoning"
    assert "thinking_blocks" not in assistant_msg


def test_hosted_vllm_redacted_thinking_blocks_are_skipped():
    """
    redacted_thinking carries opaque data with no text form; emitting it would send garbage
    into the prompt, and its presence must not shadow the real thinking blocks either.
    """
    assistant_msg = _transform_single_assistant_message(
        {
            "role": "assistant",
            "content": "Answer",
            "thinking_blocks": [
                {"type": "redacted_thinking", "data": "AAAABBBBCCCC"},
                {"type": "thinking", "thinking": "visible reasoning"},
            ],
        }
    )
    assert assistant_msg["reasoning"] == "visible reasoning"
    assert "AAAABBBBCCCC" not in json.dumps(assistant_msg)


def test_hosted_vllm_only_redacted_thinking_sets_no_reasoning():
    assistant_msg = _transform_single_assistant_message(
        {
            "role": "assistant",
            "content": "Answer",
            "thinking_blocks": [{"type": "redacted_thinking", "data": "AAAABBBBCCCC"}],
        }
    )
    assert "reasoning" not in assistant_msg
    assert "reasoning_content" not in assistant_msg
    assert "thinking_blocks" not in assistant_msg


def test_hosted_vllm_existing_reasoning_content_is_mirrored_into_reasoning():
    """
    vLLM 0.16 dropped the `reasoning_content` fallback, so a history that already carries
    reasoning_content (the OpenAI-style round trip) still needs `reasoning` populated.
    """
    assistant_msg = _transform_single_assistant_message(
        {
            "role": "assistant",
            "content": "Answer",
            "reasoning_content": "reasoning from a previous turn",
        }
    )
    assert assistant_msg["reasoning"] == "reasoning from a previous turn"
    assert assistant_msg["reasoning_content"] == "reasoning from a previous turn"


def test_hosted_vllm_existing_reasoning_content_wins_over_thinking_blocks():
    assistant_msg = _transform_single_assistant_message(
        {
            "role": "assistant",
            "content": "Answer",
            "reasoning_content": "authoritative reasoning",
            "thinking_blocks": [{"type": "thinking", "thinking": "stale reasoning"}],
        }
    )
    assert assistant_msg["reasoning"] == "authoritative reasoning"
    assert assistant_msg["reasoning_content"] == "authoritative reasoning"
    assert "stale reasoning" not in json.dumps(assistant_msg)


def test_hosted_vllm_assistant_without_reasoning_gains_no_reasoning_keys():
    assistant_msg = _transform_single_assistant_message({"role": "assistant", "content": "Answer"})
    assert "reasoning" not in assistant_msg
    assert "reasoning_content" not in assistant_msg


def test_hosted_vllm_anthropic_messages_bridge_preserves_thinking():
    """
    The path that actually loses reasoning in production: a /v1/messages request for a
    hosted_vllm model has no native Anthropic config, so it is bridged to chat completions.
    The bridge turns Anthropic thinking content into thinking_blocks; this asserts the
    reasoning survives all the way into the vLLM request body.
    """
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    anthropic_messages = [
        {"role": "user", "content": "9.11 or 9.8, which is greater?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Compare the decimals", "signature": "sig"},
                {"type": "text", "text": "9.8 is greater."},
            ],
        },
        {"role": "user", "content": "Why?"},
    ]
    openai_messages = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        messages=anthropic_messages
    )

    transformed = HostedVLLMChatConfig().transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=openai_messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assistant_msg = next(m for m in transformed["messages"] if m["role"] == "assistant")
    assert assistant_msg["reasoning"] == "Compare the decimals"
    assert assistant_msg["reasoning_content"] == "Compare the decimals"
    assert assistant_msg["content"] == "9.8 is greater."


def test_hosted_vllm_assistant_structured_content_is_preserved():
    config = HostedVLLMChatConfig()
    image_block = {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image.png"},
    }
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Here is the image"}, image_block],
        },
    ]

    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assistant_msg = transformed["messages"][0]
    assert assistant_msg["content"] == [
        {"type": "text", "text": "Here is the image"},
        image_block,
    ]


def test_hosted_vllm_assistant_tool_use_content_becomes_tool_calls():
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Boston"},
                }
            ],
        },
    ]

    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assistant_msg = transformed["messages"][0]
    assert assistant_msg["content"] == ""
    assert assistant_msg["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Boston"}),
            },
        }
    ]


def test_hosted_vllm_assistant_tool_use_does_not_duplicate_existing_tool_calls():
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Boston"},
                }
            ],
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Boston"}),
                    },
                }
            ],
        },
    ]

    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assistant_msg = transformed["messages"][0]
    assert assistant_msg["content"] == ""
    assert assistant_msg["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Boston"}),
            },
        }
    ]


def test_hosted_vllm_custom_tools_are_converted_to_function_tools():
    config = HostedVLLMChatConfig()
    optional_params = config.map_openai_params(
        non_default_params={
            "tools": [
                {
                    "type": "custom",
                    "custom": {
                        "name": "apply_patch",
                        "description": "Apply text patch",
                        "format": {
                            "type": "grammar",
                            "grammar": {"syntax": "lark", "definition": "start: /.*/"},
                        },
                    },
                }
            ]
        },
        optional_params={},
        model="hosted_vllm/gpt-oss-120b",
        drop_params=False,
    )

    tools = optional_params["tools"]
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "apply_patch"
    assert tools[0]["function"]["description"] == "Apply text patch"
    assert tools[0]["function"]["parameters"]["type"] == "object"
    assert "input" in tools[0]["function"]["parameters"]["properties"]


def test_hosted_vllm_custom_tools_use_top_level_input_schema():
    config = HostedVLLMChatConfig()
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    optional_params = config.map_openai_params(
        non_default_params={
            "tools": [
                {
                    "type": "custom",
                    "name": "search",
                    "description": "Search docs",
                    "input_schema": input_schema,
                }
            ]
        },
        optional_params={},
        model="hosted_vllm/gpt-oss-120b",
        drop_params=False,
    )

    tools = optional_params["tools"]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "search"
    assert tools[0]["function"]["description"] == "Search docs"
    assert tools[0]["function"]["parameters"] == input_schema
