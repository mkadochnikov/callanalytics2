#!/bin/bash

echo "🚀 Установка NLU компонентов для анализа звонков"
echo "================================================"

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.8+"
    exit 1
fi

echo "✅ Python найден: $(python3 --version)"

# Создание виртуального окружения если его нет
if [ ! -d "venv" ]; then
    echo "📦 Создание виртуального окружения..."
    python3 -m venv venv
fi

# Активация виртуального окружения
echo "🔄 Активация виртуального окружения..."
source venv/bin/activate

# Обновление pip
echo "📦 Обновление pip..."
pip install --upgrade pip

# Установка основных зависимостей
echo "📦 Установка основных зависимостей..."
pip install -r requirements.txt

# Установка Natasha и компонентов
echo "🤖 Установка Natasha NLU..."
pip install natasha>=1.6.0
pip install razdel>=0.5.0
pip install navec>=0.10.0
pip install slovnet>=0.6.0
pip install yargy>=0.16.0
pip install ipymarkup>=0.9.0
pip install pymorphy2>=0.9.1
pip install pymorphy2-dicts-ru>=2.4.417127.4579844

# Загрузка NLTK данных
echo "📚 Загрузка NLTK данных..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Опциональная установка DeepPavlov (требует больше ресурсов)
read -p "🤔 Установить DeepPavlov? (требует 4GB+ RAM, может занять время) [y/N]: " install_deeppavlov
if [[ $install_deeppavlov =~ ^[Yy]$ ]]; then
    echo "🤖 Установка DeepPavlov..."
    pip install deeppavlov>=1.3.0

    # Загрузка моделей DeepPavlov
    echo "📥 Загрузка моделей DeepPavlov..."
    python3 -c "from deeppavlov import configs, build_model; build_model(configs.classifiers.intents_snips_ru, download=True)"
    python3 -c "from deeppavlov import configs, build_model; build_model(configs.ner.ner_rus_bert, download=True)"
else
    echo "⏭️ Пропуск установки DeepPavlov"
fi

# Создание необходимых директорий
echo "📁 Создание структуры директорий..."
mkdir -p bitrix_analytics/custom_objections

# Тестирование установки
echo "🧪 Тестирование установки..."
python3 -c "
import sys
print('Python:', sys.version)

try:
    from natasha import Segmenter, Doc, NewsEmbedding, NewsMorphTagger
    print('✅ Natasha установлена')
except ImportError:
    print('❌ Ошибка импорта Natasha')

try:
    import torch
    print(f'✅ PyTorch установлен (устройство: {\"CUDA\" if torch.cuda.is_available() else \"CPU\"})')
except ImportError:
    print('❌ Ошибка импорта PyTorch')

try:
    import whisper
    print('✅ Whisper установлен')
except ImportError:
    print('❌ Ошибка импорта Whisper')

try:
    from transformers import pipeline
    print('✅ Transformers установлен')
except ImportError:
    print('❌ Ошибка импорта Transformers')

try:
    from deeppavlov import configs
    print('✅ DeepPavlov установлен')
except ImportError:
    print('⚠️ DeepPavlov не установлен (опционально)')
"

echo ""
echo "✨ Установка завершена!"
echo ""
echo "📝 Инструкции по использованию:"
echo "1. Активируйте виртуальное окружение: source venv/bin/activate"
echo "2. Запустите приложение: streamlit run main.py"
echo ""
echo "🔧 NLU анализ будет автоматически активирован при запуске"