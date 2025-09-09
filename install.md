# 🚀 Быстрая установка

## Автоматическая установка (рекомендуется)

```bash
# 1. Скачайте все файлы проекта
# 2. Запустите автоматический установщик
python install.py
```

Скрипт автоматически:
- ✅ Проверит версию Python
- ✅ Установит системные зависимости  
- ✅ Установит все Python пакеты
- ✅ Выберет правильную версию PyTorch (CPU/GPU)
- ✅ Протестирует установку
- ✅ Создаст файл .env

## Ручная установка

### 1. Создайте виртуальное окружение
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows  
venv\Scripts\activate
```

### 2. Установите зависимости поэтапно

**Основные пакеты:**
```bash
pip install streamlit requests pandas plotly beautifulsoup4 pytz numpy scipy librosa soundfile
```

**PyTorch (выберите одну команду):**

Для GPU (CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Для CPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**ИИ модели:**
```bash
pip install transformers>=4.35.0 openai-whisper>=20231117
```

### 3. Системные зависимости

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg portaudio19-dev python3-dev build-essential
```

**macOS:**
```bash
brew install ffmpeg portaudio
```

**Windows:**
- Скачайте FFmpeg с https://ffmpeg.org/download.html
- Добавьте в PATH

### 4. Настройка

Создайте файл `.env`:
```env
BITRIX_WEBHOOK_URL=https://your-domain.bitrix24.ru/rest/1/your-webhook-code
BITRIX_USERNAME=your-username
BITRIX_PASSWORD=your-password
```

### 5. Запуск

```bash
streamlit run bitrix_analytics.py
```

## Проверка установки

```python
# Проверьте в Python консоли
import torch
import transformers
import whisper
import streamlit

print(f"PyTorch: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
print("✅ Все модули загружены успешно!")
```

## Решение проблем

**Ошибки с pathlib/hashlib:**
- Эти модули встроенные в Python, не устанавливайте их через pip

**Ошибки CUDA:**
```bash
# Проверьте версию CUDA
nvidia-smi

# Переустановите PyTorch для вашей CUDA версии
pip uninstall torch torchvision torchaudio  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Ошибки памяти:**
- Используйте модель Whisper "base" вместо "medium"
- Обрабатывайте данные по дням, а не за большие периоды

**Медленная работа:**
- Убедитесь что используется GPU: `torch.cuda.is_available()` должно вернуть `True`
- Для CPU версии ожидайте дольшего времени обработки

## Первый запуск

При первом запуске:
1. Модели ИИ скачаются автоматически (~2-3GB)
2. Это займет 5-10 минут
3. В дальнейшем запуск будет быстрым

Готово! 🎉