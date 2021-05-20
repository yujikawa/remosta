import streamlit as st
from PIL import Image
from gen_checker import predict

st.title('星野源チェッカー')

file = st.file_uploader(label='ファイルアップロード', type=['jpg', 'png', 'jpeg'])
if file is not None:
  img = Image.open(file)
# img = Image.open(file)
  st.image(img, width=200, caption='診断した画像')
  result = predict(img) * 100
  if result <= 50:
    st.warning(f'残念ながら、あなたがガッキーと結婚できる確率は{result:.2f}%、恋ダンスでも踊ってろ！')
  elif result > 50:
    st.success(f'あなたがガッキーと結婚できる確率は{result:.2f}%、ワンチャンあるかも！')