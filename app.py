with tab1:
    st.markdown("### Draw a digit (0-9) below:")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Safe check for canvas image data
    if canvas_result is not None and canvas_result.image_data is not None:
        try:
            img_data = canvas_result.image_data
            if img_data.size > 0 and np.any(img_data > 0):
                # Convert canvas image data to PIL Image
                img_data_uint8 = img_data.astype('uint8')
                img = Image.fromarray(img_data_uint8).convert('L')
                
                # Resize to 28x28 for MNIST model
                processed_img = img.resize((28, 28))
                img_array = np.array(processed_img) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)
                
                # Predict
                preds = model.predict(img_array)
                pred_idx = np.argmax(preds[0])
                confidence = float(np.max(preds[0])) * 100
                
                st.markdown("---")
                st.success(f"### Predicted Digit: **{pred_idx}**")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                st.progress(int(confidence))
        except Exception as e:
            st.info("Kripya canvas par kuch draw karein prediction ke liye.")
