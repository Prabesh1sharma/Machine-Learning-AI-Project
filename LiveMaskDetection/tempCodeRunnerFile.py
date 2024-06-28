scale = 50
        width = int(frame.shape[1] * scale / 100)
        height = int(frame.shape[0] * scale / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)