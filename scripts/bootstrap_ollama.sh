#!/bin/sh

set -eu

ensure_model() {
    model_name="$1"
    expected_id="$2"

    if ! ollama show "${model_name}" >/dev/null 2>&1; then
        echo "Downloading Ollama model: ${model_name}"
        ollama pull "${model_name}"
    else
        echo "Ollama model already exists: ${model_name}"
    fi

    case "${model_name}" in
        *:*) listed_name="${model_name}" ;;
        *) listed_name="${model_name}:latest" ;;
    esac

    actual_id="$(
        ollama list |
            awk -v wanted="${listed_name}" \
                'NR > 1 && $1 == wanted { print $2; exit }'
    )"
    if [ -z "${actual_id}" ]; then
        echo "Unable to read the model ID for ${listed_name}." >&2
        exit 1
    fi

    if [ -n "${expected_id}" ]; then
        case "${actual_id}" in
            "${expected_id}"*) ;;
            *)
                echo "Model ID mismatch for ${listed_name}." >&2
                echo "Expected prefix: ${expected_id}" >&2
                echo "Actual ID:       ${actual_id}" >&2
                echo "Update the expected ID only after rebuilding affected embeddings." >&2
                exit 1
                ;;
        esac
    fi
}

ensure_model "${OLLAMA_CHAT_MODEL}" "${OLLAMA_CHAT_MODEL_ID:-}"
ensure_model "${OLLAMA_EMBEDDING_MODEL}" "${OLLAMA_EMBEDDING_MODEL_ID:-}"

ollama list
