from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image
from urllib.parse import urlparse

from explainer import explain_image, explain_text, explain_url
from modules.deepfake_image_analyzer import predict_image
from modules.nlp_analyzer import predict_text
from modules.url_analyzer import predict_url


st.set_page_config(
    page_title='Tri-Sentinel XAI',
    page_icon='🔎',
    layout='wide',
)


def _read_uploaded_image(file_handle) -> Image.Image:
    opened_image = Image.open(file_handle)
    return opened_image.convert('RGB')


def _url_has_path(url_value: str) -> bool:
    parsed = urlparse(url_value if url_value.startswith(('http://', 'https://')) else f'https://{url_value}')
    return bool(parsed.path and parsed.path not in {'/'})


st.title('Tri-Sentinel XAI')
st.caption('URL, NLP, deepfake image inference, and explainability for social engineering detection.')

tab_url, tab_text, tab_image = st.tabs(['URL', 'Text', 'Image'])

with tab_url:
    st.subheader('URL analysis')
    url_value = st.text_input('Enter a URL', placeholder='https://example.com/login')
    st.caption('For best results, enter the full URL including the path, such as https://claude.ai/chat/.')
    if st.button('Analyze URL', type='primary', key='analyze_url'):
        if not url_value.strip():
            st.warning('Enter a URL first.')
        else:
            with st.spinner('Running URL prediction and SHAP explanation...'):
                cleaned_url = url_value.strip()
                if not _url_has_path(cleaned_url):
                    st.warning(
                        'This detector is more reliable with the full URL path. Example: https://claude.ai/chat/'
                    )
                prediction = predict_url(cleaned_url)
                url_info = explain_url(cleaned_url)
            st.json(prediction)
            st.subheader('Responsible AI explanation')
            if url_info.get('prediction'):
                st.json(url_info['prediction'])
            if url_info.get('base_value') is not None:
                st.caption(f"SHAP base value: {float(url_info['base_value']):.4f}")
            if url_info.get('summary'):
                st.write(url_info['summary'])
            positive_contributions = url_info.get('positive_contributions', [])
            negative_contributions = url_info.get('negative_contributions', [])
            if positive_contributions or negative_contributions:
                left_column, right_column = st.columns(2)
                with left_column:
                    st.markdown('Most support phishing')
                    st.dataframe(pd.DataFrame(positive_contributions), use_container_width=True, hide_index=True)
                with right_column:
                    st.markdown('Most support legitimacy')
                    st.dataframe(pd.DataFrame(negative_contributions), use_container_width=True, hide_index=True)
            else:
                st.info('No feature contributions available for this URL.')

with tab_text:
    st.subheader('Message analysis')
    text_value = st.text_area('Enter a message', height=180, placeholder='Paste the email or SMS text here')
    if st.button('Analyze text', type='primary', key='analyze_text'):
        if not text_value.strip():
            st.warning('Enter some text first.')
        else:
            with st.spinner('Running text prediction and LIME explanation...'):
                cleaned_text = text_value.strip()
                prediction = predict_text(cleaned_text)
                text_info = explain_text(cleaned_text)
            st.json(prediction)
            st.subheader('Responsible AI explanation')
            if text_info.get('summary'):
                st.write(text_info['summary'])
            st.caption(
                f"Spam probability: {text_info['probabilities']['spam_prob']:.2f}% | "
                f"Legitimate probability: {text_info['probabilities']['legit_prob']:.2f}%"
            )
            positive_contributions = text_info.get('positive_contributions', [])
            negative_contributions = text_info.get('negative_contributions', [])
            if positive_contributions or negative_contributions:
                left_column, right_column = st.columns(2)
                with left_column:
                    st.markdown('Words supporting the predicted class')
                    st.dataframe(pd.DataFrame(positive_contributions), use_container_width=True, hide_index=True)
                with right_column:
                    st.markdown('Words arguing against the predicted class')
                    st.dataframe(pd.DataFrame(negative_contributions), use_container_width=True, hide_index=True)
            else:
                st.info('No token-level explanation was generated for this message.')

with tab_image:
    st.subheader('Image analysis')
    selected_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg', 'webp'])
    if selected_file is not None:
        preview_image = _read_uploaded_image(selected_file)
        st.image(preview_image, caption=selected_file.name, use_column_width=True)

        if st.button('Analyze image', type='primary', key='analyze_image'):
            with st.spinner('Running image prediction and Grad-CAM explanation...'):
                try:
                    prediction = predict_image(preview_image)
                    image_info = explain_image(preview_image)
                    st.json(prediction)
                    st.subheader('Responsible AI explanation')
                    if image_info.get('summary'):
                        st.write(image_info['summary'])
                    if image_info.get('heatmap'):
                        st.image(image_info['heatmap'], caption='Grad-CAM heatmap', use_column_width=True)
                    st.caption(f"Target layer: {image_info['target_layer']}")
                except Exception as e:
                    st.error(f'Image analysis failed on this load. Please refresh and try again.')
                    st.caption(f'Technical detail: {str(e)}')