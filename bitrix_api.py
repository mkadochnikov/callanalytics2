#!/usr/bin/env python3
"""
API методы для работы с Bitrix24
"""

from config import *


class BitrixAPI:
    """Класс для работы с API Bitrix24"""

    def __init__(self):
        self.webhook_url = os.getenv('BITRIX_WEBHOOK_URL', '').rstrip('/')
        self.username = os.getenv('BITRIX_USERNAME', '')
        self.password = os.getenv('BITRIX_PASSWORD', '')

        self.session = requests.Session()
        self.authenticated = False

    def make_api_call(self, method: str, params: Dict = None) -> Optional[Dict]:
        """Выполняет API вызов к Bitrix24"""
        if not self.webhook_url:
            return None

        url = f"{self.webhook_url}/{method}"

        try:
            if params is None:
                params = {}

            response = requests.post(url, json=params, timeout=30)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Ошибка API {method}: {e}")
            return None

    def authenticate_bitrix(self) -> bool:
        """Авторизуется в Bitrix24"""
        if not self.username or not self.password:
            return False

        logger.info("Авторизация в Bitrix24...")

        base_url = self.webhook_url.split('/rest/')[0]
        auth_url = f"{base_url}/auth/"

        try:
            auth_page = self.session.get(auth_url, timeout=30)

            if auth_page.status_code == 200:
                auth_data = {
                    'USER_LOGIN': self.username,
                    'USER_PASSWORD': self.password
                }

                # Парсим скрытые поля
                soup = BeautifulSoup(auth_page.content, 'html.parser')
                for hidden_input in soup.find_all('input', type='hidden'):
                    name = hidden_input.get('name')
                    value = hidden_input.get('value')
                    if name and value:
                        auth_data[name] = value

                login_response = self.session.post(auth_url, data=auth_data, timeout=30)

                if login_response.status_code == 200:
                    if 'logout' in login_response.text.lower() or 'выйти' in login_response.text.lower():
                        logger.info("Авторизация успешна!")
                        self.authenticated = True
                        return True

            return False

        except Exception as e:
            logger.error(f"Ошибка авторизации: {e}")
            return False

    def get_all_calls_for_day(self, target_day: datetime.datetime) -> List[Dict]:
        """Получает все звонки за день"""
        all_calls = []
        start = 0

        tz = pytz.timezone(PORTAL_TIMEZONE)
        start_date_utc = tz.localize(target_day).astimezone(pytz.utc)
        end_date_utc = tz.localize(target_day.replace(hour=23, minute=59, second=59)).astimezone(pytz.utc)

        logger.info(f"Загрузка звонков за {target_day.strftime('%d.%m.%Y')}...")

        while True:
            params = {
                'filter': {
                    '>=CALL_START_DATE': start_date_utc.isoformat(),
                    '<=CALL_START_DATE': end_date_utc.isoformat()
                },
                'start': start
            }

            data = self.make_api_call("voximplant.statistic.get", params)

            if data and 'result' in data and data['result']:
                batch_calls = data['result']
                all_calls.extend(batch_calls)

                if len(batch_calls) < 50:
                    break
                start += 50
            else:
                break

        logger.info(f"Найдено {len(all_calls)} звонков")
        return all_calls

    def get_user_names(self, user_ids: set) -> Dict[str, str]:
        """Получает имена пользователей"""
        if not user_ids:
            return {}

        data = self.make_api_call("user.get", {'ID': list(user_ids)})

        if data and 'result' in data:
            user_names = {}
            for user in data['result']:
                name = f"{user.get('NAME', '')} {user.get('LAST_NAME', '')}".strip()
                user_names[user['ID']] = name or f"User_{user['ID']}"
            return user_names

        return {}

    def determine_call_direction(self, call: Dict) -> str:
        """Улучшенное определение направления звонка"""
        call_type = call.get('CALL_TYPE', '')

        # Bitrix24 коды направлений
        if call_type == 1 or call_type == '1':
            return 'incoming'
        elif call_type == 2 or call_type == '2':
            return 'outgoing'

        # Дополнительная логика по полям
        call_direction = call.get('CALL_DIRECTION', '').lower()
        if 'in' in call_direction:
            return 'incoming'
        elif 'out' in call_direction:
            return 'outgoing'

        # Анализ по времени звонка
        phone_number = call.get('PHONE_NUMBER', '')
        portal_user_id = call.get('PORTAL_USER_ID', '')

        if phone_number and portal_user_id:
            call_start_date = call.get('CALL_START_DATE', '')
            if call_start_date:
                try:
                    dt = datetime.datetime.fromisoformat(call_start_date.replace('Z', '+00:00'))
                    hour = dt.hour
                    if 9 <= hour <= 18:  # Рабочие часы
                        return 'incoming'
                    else:
                        return 'outgoing'
                except:
                    pass

        return 'unknown'