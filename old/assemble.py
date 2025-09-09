#!/usr/bin/env python3
"""
Скрипт для сборки всех модулей в один файл script.py
"""


def create_complete_script():
    """Создает полный script.py из всех модулей"""

    files_to_combine = [
        "config.py",
        "ai_analyzer.py",
        "bitrix_api.py",
        "audio_processor.py",
        "data_manager.py",
        "main_analyzer.py",
        "streamlit_app.py"
    ]

    print("🔧 Создание полного script.py из модулей...")

    combined_content = []

    # Добавляем заголовок
    combined_content.append('#!/usr/bin/env python3')
    combined_content.append('"""')
    combined_content.append('Система анализа звонков Bitrix24 с ЛОКАЛЬНЫМИ моделями ИИ')
    combined_content.append('Полный цикл: загрузка -> транскрибация (Whisper) -> анализ (локальная модель) -> отчет')
    combined_content.append('"""')
    combined_content.append('')

    processed_imports = set()

    for file_name in files_to_combine:
        try:
            print(f"📄 Обрабатываем {file_name}...")

            with open(file_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Пропускаем shebang и docstring в модулях
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('from config import') or \
                        line.strip().startswith('from ai_analyzer import') or \
                        line.strip().startswith('from bitrix_api import') or \
                        line.strip().startswith('from audio_processor import') or \
                        line.strip().startswith('from data_manager import') or \
                        line.strip().startswith('from main_analyzer import'):
                    start_idx = i + 1
                    continue

                if line.strip().startswith('import ') or \
                        line.strip().startswith('from ') and 'import' in line:
                    if line.strip() not in processed_imports:
                        if file_name == "config.py":  # Импорты только из config.py
                            combined_content.append(line.rstrip())
                            processed_imports.add(line.strip())
                elif line.strip().startswith('class ') or \
                        line.strip().startswith('def ') or \
                        (line.strip() and not line.startswith('#') and not line.startswith(
                            '"""') and not line.startswith("'''")):
                    # Добавляем код класса/функции
                    combined_content.extend([line.rstrip() for line in lines[i:]])
                    break

        except FileNotFoundError:
            print(f"⚠️ Файл {file_name} не найден, пропускаем...")
            continue

        combined_content.append('')  # Разделитель между файлами

    # Записываем объединенный файл
    with open('script.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(combined_content))

    print("✅ Файл script.py успешно создан!")
    print(f"📊 Всего строк: {len(combined_content)}")

    # Проверяем синтаксис
    try:
        with open('script.py', 'r', encoding='utf-8') as f:
            compile(f.read(), 'script.py', 'exec')
        print("✅ Синтаксис проверен - ошибок нет!")
    except SyntaxError as e:
        print(f"❌ Синтаксическая ошибка: {e}")
        print(f"   Строка {e.lineno}: {e.text}")

    return True


if __name__ == "__main__":
    print("🚀 Запуск сборки полного script.py")
    print("=" * 50)

    success = create_complete_script()

    if success:
        print("\n" + "=" * 50)
        print("🎉 Готово! Теперь запустите:")
        print("   streamlit run script.py")
        print("\n💡 Файлы созданы:")
        print("   ├── script.py (основной файл)")
        print("   ├── config.py (конфигурация)")
        print("   ├── ai_analyzer.py (ИИ анализ)")
        print("   ├── bitrix_api.py (API Bitrix24)")
        print("   ├── audio_processor.py (обработка аудио)")
        print("   ├── data_manager.py (управление данными)")
        print("   ├── main_analyzer.py (основной анализатор)")
        print("   └── streamlit_app.py (веб-интерфейс)")
    else:
        print("❌ Ошибка при создании script.py")