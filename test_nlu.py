#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы NLU анализа
Запуск: python test_nlu.py
"""

import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from ai_analyzer import LocalAIAnalyzer
import json

# Тестовые транскрипции из ваших примеров
TEST_TRANSCRIPTS = {
    "Возражение по цене": """
        Добрый день, по отель дворец Нарзанов. Меня зовут Ольга. Чем могу помочь? 
        Здравствуйте, меня зовут Галина. Я хотела узнать стоимость на ноябрь. 
        Смотрите, номер делюкс без балкона за 12 дней получается 191 340. 
        Это очень дорого. Честно говоря, 200 с лишним за Кисловодск это слишком. 
        На Мальдивы дешевле слетать можно.
    """,

    "Нужно согласовать": """
        Смотрите, получается у нас 9 ночей с питанием шведский стол.
        Стоимость 230 600 рублей.
        Я все поняла. Мы обмозгуем, побеседуем с мужем и если что я вам перезвоню.
        Нужно с семьей обговорить.
    """,

    "Успешное бронирование": """
        Давайте забронируем с 3 января по 10 января, двое взрослых.
        Хорошо, договорились! Счет выставлен, ждем вас.
        Отлично, внесу предоплату сегодня.
    """,

    "Запрос информации": """
        Здравствуйте, я хотела узнать по поводу лечебных программ.
        Какие процедуры входят? Есть ли у вас профиль по урологии?
        И еще интересует можно ли с детьми приехать.
    """,

    "Сомнения в медбазе": """
        А у вас есть врач офтальмолог? Мне нужно глаза подлечить.
        А какая квалификация у врачей? Есть ли современное оборудование?
        В Пикете была хорошая база, а у вас как?
    """
}


def test_nlu_analysis():
    """Тестирует NLU анализ на примерах"""
    print("=" * 80)
    print("🧪 ТЕСТИРОВАНИЕ NLU АНАЛИЗА ТРАНСКРИПЦИЙ")
    print("=" * 80)

    # Инициализация анализатора
    print("\n📦 Инициализация анализатора...")
    try:
        analyzer = LocalAIAnalyzer()
        print("✅ Анализатор инициализирован")

        # Проверяем наличие NLU
        if hasattr(analyzer, 'use_nlu') and analyzer.use_nlu:
            print("✅ NLU модуль активен")
        else:
            print("⚠️ NLU модуль не активен, используется базовый анализ")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return

    print("\n" + "=" * 80)

    # Тестируем каждую транскрипцию
    for test_name, transcript in TEST_TRANSCRIPTS.items():
        print(f"\n📝 ТЕСТ: {test_name}")
        print("-" * 40)

        try:
            # Анализируем
            if hasattr(analyzer, 'analyze_full_dialogue'):
                result = analyzer.analyze_full_dialogue(transcript)
            else:
                # Fallback на базовые методы
                result = {
                    "topic": analyzer.classify_topic(transcript),
                    "objection": analyzer.find_objection_reason(transcript),
                    "key_points": analyzer.extract_key_points(transcript)
                }

            # Выводим результаты
            print(f"📊 Тема: {result.get('topic', 'Не определена')}")

            # Возражения
            if result.get('objections'):
                print(f"🚫 Возражений найдено: {len(result['objections'])}")
                for obj in result['objections'][:2]:
                    print(f"   • {obj.get('type', 'Неизвестное')}")
                    print(f"     → {obj.get('recommendation', 'Нет рекомендации')}")
            elif result.get('objection_reason'):
                print(f"🚫 Возражение: {result['objection_reason']}")
                if result.get('objection_recommendation'):
                    print(f"   → {result['objection_recommendation']}")
            else:
                print("✅ Возражений не обнаружено")

            # Результат звонка
            if result.get('call_result'):
                call_result = result['call_result']
                print(f"📞 Результат: {call_result.get('result', 'Не определен')}")
                if call_result.get('confidence'):
                    print(f"   Уверенность: {call_result['confidence']:.2f}")

            # Бизнес-данные
            if result.get('business_data'):
                business = result['business_data']
                if business.get('dates', {}).get('check_in'):
                    print(f"📅 Даты: найдены")
                if business.get('financial', {}).get('quoted_prices'):
                    print(f"💰 Цены: {len(business['financial']['quoted_prices'])} упоминаний")
                if business.get('guests', {}).get('total'):
                    print(f"👥 Гостей: {business['guests']['total']}")

            # Ключевые моменты
            if result.get('key_points'):
                print(f"🔑 Ключевых моментов: {len(result['key_points'])}")
                for point in result['key_points'][:2]:
                    print(f"   • {point[:70]}...")

        except Exception as e:
            print(f"❌ Ошибка анализа: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ Тестирование завершено")
    print("=" * 80)


def test_entity_extraction():
    """Тестирует извлечение сущностей"""
    print("\n" + "=" * 80)
    print("🔍 ТЕСТ ИЗВЛЕЧЕНИЯ СУЩНОСТЕЙ")
    print("=" * 80)

    test_text = """
    Здравствуйте, Ольга Яценко. Я хочу забронировать номер делюкс с балконом 
    с 15 октября по 22 октября. Нас будет двое взрослых и один ребенок 9 лет.
    Стоимость 235 980 рублей меня устраивает. Мой телефон +7 926 151 88 96,
    почта test@example.com. Нужен трансфер из Минеральных Вод.
    """

    try:
        from advanced_nlu import AdvancedNLUAnalyzer
        nlu = AdvancedNLUAnalyzer()

        # Извлекаем сущности
        result = nlu.analyze_dialogue(test_text)
        entities = result.get('entities', {})

        print("\n📝 Найденные сущности:")

        if entities.get('persons'):
            print(f"👤 Персоны: {entities['persons']}")

        if entities.get('dates'):
            print(f"📅 Даты: {len(entities['dates'])} найдено")
            for date in entities['dates']:
                print(f"   • {date.get('text', 'Неизвестно')}")

        if entities.get('money'):
            print(f"💰 Суммы:")
            for money in entities['money']:
                print(f"   • {money.get('amount', 'Неизвестно')} {money.get('currency', 'RUB')}")

        if entities.get('phones'):
            print(f"📱 Телефоны: {entities['phones']}")

        if entities.get('emails'):
            print(f"📧 Email: {entities['emails']}")

        if entities.get('locations'):
            print(f"📍 Локации: {entities['locations']}")

        if entities.get('room_types'):
            print(f"🏨 Типы номеров: {entities['room_types']}")

        if entities.get('duration_nights'):
            print(f"🌙 Продолжительность: {entities['duration_nights']} ночей")

        if entities.get('guest_count'):
            print(f"👥 Количество гостей: {entities['guest_count']}")

    except ImportError:
        print("⚠️ NLU модуль не установлен")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 Запуск тестов NLU анализа\n")

    # Тест основного анализа
    test_nlu_analysis()

    # Тест извлечения сущностей
    test_entity_extraction()

    print("\n✅ Все тесты завершены\n")