import streamlit as st
import cv2
import numpy as np
from matplotlib.animation import FuncAnimation
import tempfile
import matplotlib.pyplot as plt

def main():
    st.set_page_config(page_title="DALL-E Mini Image Generator", page_icon=":camera:", layout="wide")
    st.title("DALL-E Mini Image Generator")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(uploaded_file.read())
            file_path = temp.name
        img = cv2.imread(file_path)
        st.image(img, width=512)
        if st.button("Apply Inpainting"):
            # Apply inpainting (image completion)
            mask = np.zeros(img.shape[:2], np.uint8)
            inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            st.image(inpainted, width=512)

    if st.button("Create Animation"):
        # Create an empty figure
        fig, ax = plt.subplots()
        xdata, ydata = [], []
        ln, = plt.plot([], [], 'ro', animated=True)

        def init():
            ax.set_xlim(0, 2*np.pi)
            ax.set_ylim(-1, 1)
            return ln,

        def update(frame):
            xdata.append(frame)
            ydata.append(np.sin(frame))
            ln.set_data(xdata, ydata)
            return ln,

        ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                            init_func=init, blit=True)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
