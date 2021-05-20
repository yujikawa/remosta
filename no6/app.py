import streamlit as st
from PIL import Image
from gen_checker import predict

st.title('星野源チェッカー')

file = st.file_uploader(label='ファイルアップロード', type=['jpg', 'png', 'jpeg'])
if file is not None:
  img = Image.open(file)
# img = Image.open(file)
  st.image(img)
  result = predict(img) * 100
  if result <= 0.5:
    st.warning(f'あなたがガッキーと結婚できる確率は{result:.2f}%、諦めてチキンラーメンでも食べといて')
  elif result > 0.5:
    st.success(f'あなたがガッキーと結婚できる確率は{result:.2f}%、ワンチャンあるかも！')