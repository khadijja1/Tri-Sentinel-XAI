from __future__ import annotations

import html as html_lib
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlparse

import joblib
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.compose import ColumnTransformer


BASE_DIR = Path(__file__).resolve().parent.parent  
MODEL_PATH = BASE_DIR / 'Models' / 'url_xgb.ubj'
PREPROCESSOR_PATH = BASE_DIR / 'Models' / 'url_preprocessor.pkl'
FEATURES_PATH = BASE_DIR / 'Models' / 'url_feature_columns.pkl'
REQUEST_TIMEOUT_SECONDS = 12
USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
)


model = xgb.XGBClassifier()
preprocessor: ColumnTransformer
feature_cols: list[str] = []


def _load_artifacts() -> tuple[ColumnTransformer, list[str]]:
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError(
            'URL model artifacts are missing. Run the archived model export first.'
        )

    model.load_model(MODEL_PATH)
    preprocessor_artifact = joblib.load(PREPROCESSOR_PATH)
    feature_columns_artifact = joblib.load(FEATURES_PATH)
    return preprocessor_artifact, feature_columns_artifact


preprocessor, feature_cols = _load_artifacts()


def _build_tld_legitimacy_map() -> dict[str, float]:
    dataset_path = BASE_DIR / 'datasets' / 'PhiUSIIL_Phishing_URL_Dataset.csv'
    if not dataset_path.exists():
        return {} 
    data = pd.read_csv(dataset_path, usecols=['TLD', 'TLDLegitimateProb'])
    grouped = data.groupby(data['TLD'].astype(str))['TLDLegitimateProb'].mean()
    return {str(key).lower(): float(value) for key, value in grouped.items()}


TLD_LEGITIMATE_PROB_MAP = _build_tld_legitimacy_map()
DEFAULT_TLD_LEGITIMATE_PROB = float(np.mean(list(TLD_LEGITIMATE_PROB_MAP.values()))) if TLD_LEGITIMATE_PROB_MAP else 0.0


def _extract_tld(domain: str) -> str:
    if '.' not in domain:
        return 'unknown'
    return domain.split('.')[-1].lower()


def _normalize_host(url: str) -> str:
    parsed = urlparse(url if url.startswith(('http://', 'https://')) else f'http://{url}')
    return parsed.netloc.lower().split(':')[0].removeprefix('www.')


@lru_cache(maxsize=128)
def _fetch_page(url: str) -> tuple[str, str]:
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=True,
            headers={'User-Agent': USER_AGENT},
        )
        response.raise_for_status()
        return response.url, response.text
    except requests.RequestException:
        return url, ''


def _strip_html_to_text(html_text: str) -> str:
    without_scripts = re.sub(r'<script\b.*?</script>', ' ', html_text, flags=re.I | re.S)
    without_styles = re.sub(r'<style\b.*?</style>', ' ', without_scripts, flags=re.I | re.S)
    without_tags = re.sub(r'<[^>]+>', ' ', without_styles)
    return re.sub(r'\s+', ' ', html_lib.unescape(without_tags)).strip()


def _extract_first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.I | re.S)
    if not match:
        return ''
    return html_lib.unescape(match.group(1)).strip()


def _tokenize_text(text: str) -> set[str]:
    return {token for token in re.split(r'[^a-z0-9]+', text.lower()) if len(token) > 1}


def _page_link_targets(html_text: str) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for attr_name in ('href', 'src', 'action'):
        pattern = rf'{attr_name}\s*=\s*["\']([^"\']+)["\']'
        for match in re.finditer(pattern, html_text, flags=re.I | re.S):
            targets.append((attr_name, html_lib.unescape(match.group(1)).strip()))
    return targets


def _is_external_target(target: str, base_host: str, base_url: str) -> bool:
    if not target or target.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
        return False
    absolute_target = target if target.startswith(('http://', 'https://')) else urljoin(base_url, target)
    parsed = urlparse(absolute_target)
    if not parsed.netloc:
        return False
    target_host = parsed.netloc.lower().split(':')[0].removeprefix('www.')
    return target_host != base_host and not target_host.endswith(f'.{base_host}') and not base_host.endswith(f'.{target_host}')


def _count_targets(html_text: str, base_host: str, base_url: str) -> tuple[int, int, int]:
    self_refs = 0
    external_refs = 0
    empty_refs = 0
    for _, target in _page_link_targets(html_text):
        if not target or target.startswith(('#', 'javascript:void(0)', 'javascript:;', 'mailto:', 'tel:')):
            empty_refs += 1
            continue
        absolute_target = target if target.startswith(('http://', 'https://')) else urljoin(base_url, target)
        parsed = urlparse(absolute_target)
        if not parsed.netloc:
            self_refs += 1
            continue
        target_host = parsed.netloc.lower().split(':')[0].removeprefix('www.')
        if target_host == base_host or target_host.endswith(f'.{base_host}') or base_host.endswith(f'.{target_host}'):
            self_refs += 1
        else:
            external_refs += 1
    return self_refs, empty_refs, external_refs


def _count_external_forms(html_text: str, base_host: str, base_url: str) -> int:
    count = 0
    for form_match in re.finditer(r'<form\b[^>]*>', html_text, flags=re.I | re.S):
        form_tag = form_match.group(0)
        action_match = re.search(r'action\s*=\s*["\']([^"\']*)["\']', form_tag, flags=re.I | re.S)
        action = html_lib.unescape(action_match.group(1)).strip() if action_match else ''
        if action and _is_external_target(action, base_host, base_url):
            count += 1
    return count


def _count_redirects(html_text: str, base_host: str, base_url: str) -> tuple[int, int]:
    redirect_targets: list[str] = []
    redirect_targets.extend(re.findall(r'http-equiv\s*=\s*["\']refresh["\'][^>]*content\s*=\s*["\'][^"\']*url=([^"\';>]+)', html_text, flags=re.I | re.S))
    redirect_targets.extend(re.findall(r'location\.(?:href|replace)\s*[:=]\s*["\']([^"\']+)["\']', html_text, flags=re.I | re.S))
    redirect_targets.extend(re.findall(r'window\.location\s*[:=]\s*["\']([^"\']+)["\']', html_text, flags=re.I | re.S))

    total = len(redirect_targets)
    self_redirects = 0
    for target in redirect_targets:
        if not _is_external_target(target, base_host, base_url):
            self_redirects += 1
    return total, self_redirects


def _similarity_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return round(100.0 * len(left & right) / max(len(right), 1), 2)


def _build_page_feature_values(url: str, final_url: str, html_text: str) -> dict[str, float | int]:
    base_host = _normalize_host(url)
    final_host = _normalize_host(final_url)
    title_text = _extract_first_match(r'<title[^>]*>(.*?)</title>', html_text)
    description_text = _extract_first_match(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)["\']', html_text)
    text_content = _strip_html_to_text(html_text)
    domain_tokens = _tokenize_text(base_host)
    title_tokens = _tokenize_text(title_text)
    url_tokens = _tokenize_text(urlparse(url).path + ' ' + urlparse(url).netloc)

    self_refs, empty_refs, external_refs = _count_targets(html_text, final_host, final_url)
    redirect_count, self_redirect_count = _count_redirects(html_text, final_host, final_url)

    return {
        'LineOfCode': len(html_text.splitlines()) if html_text else 0,
        'LargestLineLength': max((len(line) for line in html_text.splitlines()), default=0),
        'HasTitle': int(bool(title_text)),
        'DomainTitleMatchScore': _similarity_score(domain_tokens, title_tokens),
        'URLTitleMatchScore': _similarity_score(url_tokens, title_tokens),
        'HasFavicon': int(bool(re.search(r'<link[^>]+rel=["\'][^"\']*icon[^"\']*["\']', html_text, flags=re.I))),
        'Robots': int(bool(re.search(r'<meta[^>]+name=["\']robots["\']', html_text, flags=re.I))),
        'IsResponsive': int(bool(re.search(r'<meta[^>]+name=["\']viewport["\']', html_text, flags=re.I))),
        'NoOfURLRedirect': int(redirect_count),
        'NoOfSelfRedirect': int(self_redirect_count),
        'HasDescription': int(bool(description_text)),
        'NoOfPopup': len(re.findall(r'window\.open\s*\(|alert\s*\(|prompt\s*\(', html_text, flags=re.I)),
        'NoOfiFrame': len(re.findall(r'<iframe\b', html_text, flags=re.I)),
        'HasExternalFormSubmit': int(_count_external_forms(html_text, final_host, final_url) > 0),
        'HasSocialNet': int(bool(re.search(r'facebook\.com|twitter\.com|x\.com|instagram\.com|linkedin\.com|youtube\.com|tiktok\.com', html_text, flags=re.I))),
        'HasSubmitButton': len(re.findall(r'<input[^>]+type=["\']submit["\']|<button[^>]+type=["\']submit["\']', html_text, flags=re.I)),
        'HasHiddenFields': len(re.findall(r'<input[^>]+type=["\']hidden["\']', html_text, flags=re.I)),
        'HasPasswordField': len(re.findall(r'<input[^>]+type=["\']password["\']', html_text, flags=re.I)),
        'HasCopyrightInfo': int('copyright' in text_content.lower() or '©' in html_text),
        'NoOfImage': len(re.findall(r'<img\b', html_text, flags=re.I)),
        'NoOfCSS': len(re.findall(r'<link\b[^>]+rel=["\'][^"\']*stylesheet[^"\']*["\']', html_text, flags=re.I)) + len(re.findall(r'<style\b', html_text, flags=re.I)),
        'NoOfJS': len(re.findall(r'<script\b', html_text, flags=re.I)),
        'NoOfSelfRef': int(self_refs),
        'NoOfEmptyRef': int(empty_refs),
        'NoOfExternalRef': int(external_refs),
    }


def _build_feature_frame(url: str) -> pd.DataFrame:
    parsed = urlparse(url if url.startswith(('http://', 'https://')) else f'http://{url}')
    domain = parsed.netloc
    tld = _extract_tld(domain)
    final_url, html_text = _fetch_page(url)

    url_len = len(url)
    no_letters = sum(character.isalpha() for character in url)
    no_digits = sum(character.isdigit() for character in url)
    obfuscated_chars = re.findall(r'%[0-9a-fA-F]{2}', url)
    no_obfuscated = len(obfuscated_chars)

    features = {
        'URLLength': url_len,
        'DomainLength': len(domain),
        'IsDomainIP': int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain))),
        'TLD': tld,
        'CharContinuationRate': len(re.findall(r'[a-zA-Z]{3,}', url)) / max(url_len, 1),
        'TLDLegitimateProb': TLD_LEGITIMATE_PROB_MAP.get(tld, DEFAULT_TLD_LEGITIMATE_PROB),
        'URLCharProb': no_letters / max(url_len, 1),
        'TLDLength': len(tld),
        'NoOfSubDomain': max(0, domain.count('.') - 1),
        'HasObfuscation': int(no_obfuscated > 0),
        'NoOfObfuscatedChar': no_obfuscated,
        'ObfuscationRatio': no_obfuscated / max(url_len, 1),
        'NoOfLettersInURL': no_letters,
        'LetterRatioInURL': no_letters / max(url_len, 1),
        'NoOfDegitsInURL': no_digits,
        'DegitRatioInURL': no_digits / max(url_len, 1),
        'NoOfEqualsInURL': url.count('='),
        'NoOfQMarkInURL': url.count('?'),
        'NoOfAmpersandInURL': url.count('&'),
        'NoOfOtherSpecialCharsInURL': sum(character in '@!#$%^*()_+[]{}|;:,<>' for character in url),
        'SpacialCharRatioInURL': sum(not character.isalnum() for character in url) / max(url_len, 1),
        'IsHTTPS': int(url.startswith('https')),
        'Bank': int(bool(re.search(r'bank|paypal|payment', url, re.I))),
        'Pay': int(bool(re.search(r'pay|checkout|billing', url, re.I))),
        'Crypto': int(bool(re.search(r'crypto|bitcoin|wallet', url, re.I))),
    }

    features.update(_build_page_feature_values(url, final_url, html_text))

    frame = pd.DataFrame([features], columns=feature_cols)
    frame['TLD'] = frame['TLD'].fillna('unknown').astype(object)

    numeric_cols = [column for column in feature_cols if column != 'TLD']
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors='coerce').fillna(0.0).astype(np.float64)

    return frame


def _page_signal_scores(frame: pd.DataFrame) -> tuple[int, int]:
    row = frame.iloc[0]

    legit_signals = sum(
        [
            bool(row['HasTitle']),
            bool(row['HasDescription']),
            bool(row['HasFavicon']),
            bool(row['IsResponsive']),
            bool(row['Robots']),
            bool(row['HasCopyrightInfo']),
            float(row['DomainTitleMatchScore']) >= 60.0,
            float(row['URLTitleMatchScore']) >= 60.0,
            float(row['NoOfSelfRef']) >= float(row['NoOfExternalRef']),
            int(row['HasExternalFormSubmit']) == 0,
            int(row['NoOfPopup']) == 0,
        ]
    )

    phish_signals = sum(
        [
            not bool(row['HasTitle']),
            not bool(row['HasDescription']),
            not bool(row['HasFavicon']),
            float(row['NoOfExternalRef']) > float(row['NoOfSelfRef']) * 2.0,
            int(row['HasExternalFormSubmit']) > 0,
            int(row['NoOfPopup']) > 0,
            float(row['DomainTitleMatchScore']) < 25.0 and bool(row['HasTitle']),
            float(row['URLTitleMatchScore']) < 25.0 and bool(row['HasTitle']),
            int(row['NoOfEmptyRef']) > 10,
        ]
    )

    return legit_signals, phish_signals


def extract_features_from_url(url: str) -> pd.DataFrame:
    return _build_feature_frame(url)


def predict_url(url: str) -> dict:
    raw = extract_features_from_url(url)
    transformed = preprocessor.transform(raw)
    probabilities = model.predict_proba(transformed)[0]
    label = int(model.predict(transformed)[0])

    legit_signals, phish_signals = _page_signal_scores(raw)
    model_phishing_prob = float(probabilities[0])
    heuristic_phishing_prob = phish_signals / max(legit_signals + phish_signals, 1)
    combined_phishing_prob = 0.6 * model_phishing_prob + 0.4 * heuristic_phishing_prob

    if legit_signals >= 6 and phish_signals <= 2:
        is_phishing = False
        combined_phishing_prob = min(combined_phishing_prob, 0.35)
    elif phish_signals >= 5 and legit_signals <= 3:
        is_phishing = True
        combined_phishing_prob = max(combined_phishing_prob, 0.65)
    else:
        is_phishing = combined_phishing_prob >= 0.5

    phishing_prob = round(float(combined_phishing_prob) * 100, 2)
    legit_prob = round((1.0 - float(combined_phishing_prob)) * 100, 2)

    return {
        'url': url,
        'label': 'Phishing' if is_phishing else 'Legitimate',
        'confidence': round(float(max(1.0 - combined_phishing_prob, combined_phishing_prob)) * 100, 2),
        'phishing_prob': phishing_prob,
        'legit_prob': legit_prob,
    }


if __name__ == '__main__':
    sample_urls = [
        'https://www.bing.com/search?q=deprecated',
        'https://claude.ai/chat',
        'https://www.google.com',
        'https://www.wikipedia.org/wiki/Machine_learning',
        'https://github.com/openai/whisper',
        'http://192.168.1.1/banking/verify-account.php?user=admin',
        'http://paypa1-secure.suspicious-domain.tk/login?token=abc123',
        'http://free-iphone-winner.click/claim?id=99182&user=victim',
    ]

    print('Starting URL inference tests...')
    print('=' * 65)

    for sample_url in sample_urls:
        result = predict_url(sample_url)
        flag = '🔴' if result['label'] == 'Phishing' else '🟢'
        print(f"\n{flag}  URL        : {sample_url[:65]}")
        print(f"    Prediction : {result['label']}")
        print(f"    Confidence : {result['confidence']}%")
        print(f"    Phishing: {result['phishing_prob']}%  |  Legit: {result['legit_prob']}%")
        print('-' * 65)