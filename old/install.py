#!/usr/bin/env python3
"""
Скрипт автоматической установки всех зависимостей для Bitrix24 Local AI Analytics
"""

import subprocess
import sys
import platform
import os


def run_command(command):
    """Выполняет команду и выводит результат"""
    print(f"Выполнение: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"Предупреждения: {result.stderr}")

    return result.returncode == 0


def check_python_version():
    """Проверяет версию Python"""
    version = sys.version_info
    print(f"Python версия: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Требуется Python 3.8 или выше!")
        return False

    print("✅ Версия Python подходит")
    return True


def detect_cuda():
    """Определяет доступность CUDA"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU обнаружена")
            # Извлекаем версию CUDA
            if "CUDA Version:" in result.stdout:
                cuda_line = [line for line in result.stdout.split('\n') if 'CUDA Version:' in line]
                if cuda_line:
                    print(f"CUDA версия: {cuda_line[0].split('CUDA Version:')[1].strip()}")
            return True
        else:
            print("ℹ️ NVIDIA GPU не обнаружена, будет использоваться CPU")
            return False
    except FileNotFoundError:
        print("ℹ️ NVIDIA драйверы не найдены, будет использоваться CPU")
        return False


def install_system_dependencies():
    """Устанавливает системные зависимости"""
    print("\n=== Установка системных зависимостей ===")

    system = platform.system().lower()

    if system == "linux":
        print("Обнаружена Linux система")
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y ffmpeg portaudio19-dev python3-dev build-essential"
        ]

        for cmd in commands:
            print(f"\nВыполнение: {cmd}")
            if not run_command(cmd):
                print(f"⚠️ Команда завершилась с ошибкой: {cmd}")
                print("Попробуйте выполнить вручную или проверьте права sudo")

    elif system == "darwin":  # macOS
        print("Обнаружена macOS система")
        # Проверяем Homebrew
        if run_command("which brew"):
            commands = [
                "brew install ffmpeg portaudio"
            ]
            for cmd in commands:
                run_command(cmd)
        else:
            print("⚠️ Homebrew не найден. Установите его или установите ffmpeg вручную")

    elif system == "windows":
        print("Обнаружена Windows система")
        print("ℹ️ Для Windows установите FFmpeg вручную с https://ffmpeg.org/download.html")
        print("ℹ️ И добавьте FFmpeg в PATH")

    print("✅ Системные зависимости обработаны")


def install_python_packages():
    """Устанавливает Python пакеты"""
    print("\n=== Установка Python зависимостей ===")

    # Основные пакеты
    basic_packages = [
        "streamlit>=1.28.0",
        "requests>=2.31.0",
        "pandas>=2.0.0",
        "plotly>=5.15.0",
        "beautifulsoup4>=4.12.0",
        "pytz>=2023.3",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0"
    ]

    print("Устанавливаем основные пакеты...")
    for package in basic_packages:
        if not run_command(f"pip install '{package}'"):
            print(f"❌ Ошибка установки {package}")
            return False

    # PyTorch - отдельно, так как зависит от системы
    print("\n=== Установка PyTorch ===")
    cuda_available = detect_cuda()

    if cuda_available:
        print("Устанавливаем PyTorch с поддержкой CUDA...")
        torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        print("Устанавливаем PyTorch для CPU...")
        torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    if not run_command(torch_cmd):
        print("❌ Ошибка установки PyTorch")
        return False

    # Transformers и Whisper
    print("\n=== Установка ИИ моделей ===")
    ai_packages = [
        "transformers>=4.35.0",
        "openai-whisper>=20231117"
    ]

    for package in ai_packages:
        print(f"Устанавливаем {package}...")
        if not run_command(f"pip install '{package}'"):
            print(f"❌ Ошибка установки {package}")
            return False

    print("✅ Все Python пакеты установлены")
    return True


def test_installation():
    """Тестирует установку"""
    print("\n=== Тестирование установки ===")

    tests = [
        ("import torch; print(f'PyTorch: {torch.__version__}')", "PyTorch"),
        ("import transformers; print(f'Transformers: {transformers.__version__}')", "Transformers"),
        ("import whisper; print(f'Whisper: доступен')", "Whisper"),
        ("import streamlit; print(f'Streamlit: {streamlit.__version__}')", "Streamlit"),
        ("import torch; print(f'CUDA доступна: {torch.cuda.is_available()}')", "CUDA проверка")
    ]

    for test_code, name in tests:
        try:
            result = subprocess.run([sys.executable, "-c", test_code],
                                    capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ {name}: {result.stdout.strip()}")
            else:
                print(f"❌ {name}: {result.stderr.strip()}")
        except Exception as e:
            print(f"❌ {name}: Ошибка теста - {e}")


def create_env_file():
    """Создает пример .env файла"""
    env_content = """# Настройки Bitrix24
BITRIX_WEBHOOK_URL=https://your-domain.bitrix24.ru/rest/1/your-webhook-code
BITRIX_USERNAME=your-username  
BITRIX_PASSWORD=your-password

# Локальные модели загружаются автоматически
# Дополнительных API ключей НЕ требуется!
"""

    if not os.path.exists('.env'):
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✅ Создан файл .env с примером настроек")
    else:
        print("ℹ️ Файл .env уже существует")


def main():
    """Основная функция установки"""
    print("🤖 Установка Bitrix24 Local AI Analytics")
    print("=" * 50)

    # Проверяем Python версию
    if not check_python_version():
        return

    try:
        # Устанавливаем системные зависимости
        install_system_dependencies()

        # Устанавливаем Python пакеты
        if not install_python_packages():
            print("❌ Ошибка установки Python пакетов")
            return

        # Тестируем установку
        test_installation()

        # Создаем .env файл
        create_env_file()

        print("\n" + "=" * 50)
        print("🎉 Установка завершена!")
        print("\nСледующие шаги:")
        print("1. Отредактируйте файл .env с вашими настройками Bitrix24")
        print("2. Запустите: streamlit run bitrix_analytics.py")
        print("3. Откройте браузер по адресу: http://localhost:8501")
        print("\n💡 При первом запуске модели ИИ скачаются автоматически (~2-3GB)")

    except KeyboardInterrupt:
        print("\n❌ Установка прервана пользователем")
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()