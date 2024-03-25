import streamlit as st
import PIL
from PIL import Image, ImageOps
from main1 import predict_color #importing predicting color function
# display image with the size and rgb color
def display_image():
    img = Image.new("RGB", (200, 200), color=(Red,Green,Blue))
    img = ImageOps.expand(img, border=1, fill='black')  # border to the img
    st.image(img, caption='RGB Color')
if __name__ == "__main__":
    hide_st_style = """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    st.sidebar.title("About")
    st.sidebar.info(
            "**RGB Color Classifier** can Predict upto 11 Distinct Color Classes based on the RGB input by User from the sliders\n\n"
            "The 11 Classes are *Red, Green, Blue, Yellow, Orange, Pink, Purple, Brown, Grey, Black and White*\n\n"
            "Prediction of colors using ANN(Artificial Neutral Network)"
            )
    Title_html = """
    <style>
        .title h1{
          user-select: none;
          font: Calibri;
          font-size: 43px;
          color: white;
          background: red;
          background-size: 600vw 600vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 10s linear infinite forwards;
        }
        @keyframes slide {
          0%{
            background-position-x: 0%;
          }
          100%{
            background-position-x: 600vw;
          }
        }
    </style> 
    
    <div class="title">
        <h1>RGB Color Classifier</h1>
    </div>
    """
    st.markdown(Title_html, unsafe_allow_html=True) #Title rendering
    Red = st.slider( label="RED value: ", min_value=0, max_value=255, value=0, key="red")
    Green = st.slider(label="GREEN value: ", min_value=0, max_value=255, value=0, key="green")
    Blue = st.slider(label="BLUE value: ", min_value=0, max_value=255, value=0, key="blue")
    st.write('Red: {}, Green: {}, Blue: {}'.format(Red, Green, Blue))
    display_image()
    result = ""
    if st.button("Predict"):
        result = predict_color(Red, Green, Blue)
        st.success('The Color is {}!'.format(result))
