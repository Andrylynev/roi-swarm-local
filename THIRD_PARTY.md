# Сторонние компоненты и модели

Установка скачивает стороннее ПО и веса непосредственно из источников авторов.
Веса, runtime, Python-окружения, кэши и пользовательские настройки не включены в этот
Git-репозиторий. Лицензия приложения не заменяет лицензии этих компонентов.
Проверяйте полный текст лицензий и model cards перед использованием/распространением;
этот список — уведомление, а не юридическая консультация.

## Inference runtime

- **PrismML llama.cpp fork**: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp),
  закреплённый release `prism-b10777-e3de715`, [MIT LICENSE](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10777-e3de715/LICENSE).
  Исходный llama.cpp/ggml: copyright © 2023–2026 The ggml authors; сохраняйте LICENSE/уведомления
  из runtime-архива при распространении. Этот fork содержит native ternary kernels для
  `PQ2_0`/`PTQ1_0`; произвольный upstream llama.cpp не является совместимой заменой.
- **NVIDIA CUDA runtime** (только CUDA-путь): сторонние NVIDIA-библиотеки в архиве издателя
  Prism; применяются [условия NVIDIA CUDA Toolkit](https://docs.nvidia.com/cuda/eula/index.html).
  Наличие публичной ссылки не делает их MIT. Системный NVIDIA-драйвер пользователь
  устанавливает отдельно; этот проект не распространяет и не устанавливает драйвер.

## Базовые модели

Автор: **Prism ML**. Семейство Bonsai 2 Ternary, производное от Qwen3.5.
Официальные model cards указывают **Apache-2.0**. Сохраняйте лицензионные уведомления
авторов при самостоятельном распространении весов.

| Роль | Репозиторий | Закреплённая ревизия | Выбранный файл |
| --- | --- | --- | --- |
| Старшие роли | [prism-ml/Ternary-Bonsai-2-9B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf) | `5b2abb02362efbed46bbe204014b02837778b2c2` | `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf` |
| Balanced workers | [prism-ml/Ternary-Bonsai-2-4B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf) | `8d6557168168f91ea6697a4a49ed2ef622b53adb` | `Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf` |
| Router / fast workers | [prism-ml/Ternary-Bonsai-2-2B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf) | `683f7be2629342a50987627b40881e54761a2430` | `Ternary-Bonsai-2-2B-PQ2_0.gguf` |

Исходные семейства: [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B),
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B),
[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B).
SHA-256 и точные размеры скачиваемых файлов приведены в [`local/assets.json`](local/assets.json).
Используется текстовый путь; отдельные vision mmproj в базовую установку не входят.

## Дополнительная security-модель (`--with-security`)

- Исходный автор: [WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B](https://huggingface.co/WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B).
- Community GGUF-конвертация: [bartowski/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-GGUF](https://huggingface.co/bartowski/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-GGUF),
  ревизия `4874da7ad5fa4da4f2584ae64ddc21f5a8e7dfce`, файл
  `WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-Q4_K_M.gguf`.
- Metadata исходной модели/конвертации содержит `apache-2.0`, **но исходная model card
  дополнительно содержит «Apache-2.0 + WhiteRabbitNeo Extended Version» и Usage Restrictions**.
  В частности, запрещены незаконное использование, military use, вред несовершеннолетним,
  неправомерное раскрытие персональных данных, определённые дискриминационные и
  полностью автоматизированные юридически значимые решения. Это не исчерпывающий пересказ:
  прочитайте полный актуальный текст автора. Не характеризуйте эту модель как безусловно
  свободную от ограничений только по YAML metadata конвертации.

Загрузка отключена по умолчанию. Модель предназначена для разрешённых текстовых консультаций;
она не объявляется проверенным native tool-worker для Hermes.

## Среда исполнения и зависимости

- **Hermes Agent / Nous Research**: [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent).
  Устанавливается отдельная закреплённая setup-ом копия, а не изменяется пользовательский
  Hermes. Смотрите LICENSE и уведомления в выбранной ревизии, включая лицензии её зависимостей.
- **uv / Astral**: [astral-sh/uv](https://github.com/astral-sh/uv), MIT OR Apache-2.0;
  если uv отсутствует, bootstrap использует официальный user-scoped installer.
- **CPython / Python Software Foundation**: [Python license](https://docs.python.org/3/license.html).
  Управляемый Python может поступать из python-build-standalone; сохраняйте включённые
  уведомления и лицензии поставляемых библиотек.
- Python-зависимости приложения закреплены в [`uv.lock`](uv.lock): в том числе FastAPI,
  Uvicorn, LangGraph, OpenAI Python SDK, Pydantic, tiktoken, AnyIO, Typer, Rich, PyYAML
  и python-dotenv. У каждой зависимости собственная лицензия в установленном дистрибутиве.
  Список транзитивных зависимостей не ограничивается этим перечислением.
- Optional extra `embeddings` не устанавливается клиентским bootstrap. Его отдельные
  ONNX/модельные компоненты также требуют проверки собственных условий, если вы его включаете.

Git и curl — предварительные инструменты пользователя. Этот проект не включает их бинарники
и не устанавливает их с административными правами. Установщик не передаёт право использования
сторонних торговых марок и не обещает одобрения со стороны перечисленных авторов.
