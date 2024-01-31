import cv2
import numpy as np
import pyopencl as cl

global image_size, MAX_DISTANCE
global cap

global platforms
global device
global ctx
global commQ
global prog

MAX_ANGLE = 180
COLOR = (255, 0, 0)
THICKNESS = 2


def VideoAquisition():
    global cap
    global image_size, MAX_DISTANCE

    pathname = r"C:/Users/santo/Desktop/TAPDI/aula100/"
    filename = "video1.mp4"
    cap = cv2.VideoCapture(pathname + filename)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    MAX_DISTANCE = int(np.sqrt((width ** 2) + (height ** 2)))
    image_size = (width, height)


def BuildKernel():
    try:
        global platforms, device, ctx, commQ, prog

        platforms = cl.get_platforms()
        platform = platforms[0]

        devices = platform.get_devices()
        device = devices[0]

        ctx = cl.Context(devices)
        commQ = cl.CommandQueue(ctx, device)

        file = open("final.cl", "r")

        prog = cl.Program(ctx, file.read())
        prog.build()
    except Exception as e:
        print(e)
        return False
    return True


def SobelKernel(img):
    try:
        img_dst = np.empty_like(imageBGRA)

        memImageIn = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNSIGNED_INT8),
                              shape=(img.shape[1], img.shape[0]),  # image width, height
                              pitches=(img.strides[0], img.strides[1]),
                              hostbuf=img.data)
        memImageOut = cl.Image(ctx, cl.mem_flags.WRITE_ONLY,
                               cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNSIGNED_INT8),
                               shape=(img.shape[1], img.shape[0]))

        kernelName1 = prog.sobel_implementation
        kernelName1.set_arg(0, memImageIn)
        kernelName1.set_arg(1, memImageOut)

    except Exception as e:
        print(e)
        return False

    globalWorkSize = image_size
    workGroupSize = (16, 16)
    kernelEvent = cl.enqueue_nd_range_kernel(commQ, kernelName1,
                                             global_work_size=globalWorkSize, local_work_size=workGroupSize)

    kernelEvent.wait()

    cl.enqueue_copy(commQ, img_dst, memImageOut, origin=(0, 0, 0), region=(img.shape[1], img.shape[0], 1))

    memImageIn.release()
    memImageOut.release()
    return img_dst


def HoughKernel(img):
    try:
        lefthough_spaces = np.zeros((MAX_DISTANCE, MAX_ANGLE), dtype=np.uint32)
        righthough_spaces = np.zeros((MAX_DISTANCE, MAX_ANGLE), dtype=np.uint32)
        region_yUp = 130  # linha de baixo
        region_yDown = 100  # linha de cima

        memImageIn = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNSIGNED_INT8),
                              shape=(img.shape[1], img.shape[0]),  # image width, height
                              pitches=(img.strides[0], img.strides[1]),
                              hostbuf=img.data)

        LeftmemBufferHS = cl.Buffer(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE,
                                    hostbuf=lefthough_spaces)
        RightmemBufferHS = cl.Buffer(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE,
                                     hostbuf=righthough_spaces)

        kernelName2 = prog.hough_implementation
        kernelName2.set_arg(0, memImageIn)
        kernelName2.set_arg(1, LeftmemBufferHS)
        kernelName2.set_arg(2, RightmemBufferHS)
        kernelName2.set_arg(3, np.int32(image_size[0]))
        kernelName2.set_arg(4, np.int32(image_size[1]))
        kernelName2.set_arg(5, np.int32(region_yDown))
        kernelName2.set_arg(6, np.int32(region_yUp))
    except Exception as e:
        print(e)
        return False

    dimension = 4  # R,G,B,A
    xBlockSize = 16
    yBlockSize = 16
    xBlocksNumber = round(imageBGRA.shape[1] / xBlockSize)
    yBlocksNumber = round(imageBGRA.shape[0] / yBlockSize)
    workItemSize = (xBlockSize, yBlockSize, dimension)
    workGroupSize = (xBlocksNumber * xBlockSize, yBlocksNumber * yBlockSize, dimension)
    kernelEvent = cl.enqueue_nd_range_kernel(commQ, kernelName2, global_work_size=workGroupSize,
                                             local_work_size=workItemSize)
    kernelEvent.wait()

    cl.enqueue_copy(commQ, lefthough_spaces, LeftmemBufferHS)
    cl.enqueue_copy(commQ, righthough_spaces, RightmemBufferHS)

    memImageIn.release()
    LeftmemBufferHS.release()
    RightmemBufferHS.release()
    # print(hough_spaces)
    return lefthough_spaces, righthough_spaces


def DrawLines(img, hough_space, hough_space2):
    max_index_left = np.unravel_index(hough_space[:, :85].argmax(), hough_space[:, :85].shape)
    rho = max_index_left[0]
    theta = np.deg2rad(max_index_left[1])
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    pt1_left = (x1, y1)
    pt2_left = (x2, y2)
    cv2.line(img, (x1, y1), (x2, y2), COLOR, THICKNESS, cv2.LINE_AA)

    # Draw line through max: for theta between 95 and 180
    max_index_right = np.unravel_index(hough_space2[:, 95:].argmax(), hough_space2[:, 95:].shape)
    rho = max_index_right[0]
    theta = np.deg2rad(max_index_right[1] + 95)  # Adjust theta by adding 95 degrees
    a = -np.cos(theta)
    b = np.sin(theta)
    x0 = img.shape[1] - a * rho
    y0 = b * rho
    x1 = int(x0 - 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 + 1000 * (-b))
    y2 = int(y0 - 1000 * a)

    pt1_right = (x1, y1)
    pt2_right = (x2, y2)
    cv2.line(img, (x1, y1), (x2, y2), COLOR, THICKNESS, cv2.LINE_AA)

    return img, pt1_left, pt2_left, pt1_right, pt2_right


def detect_cars(img, yolo_net, pt1_left, pt2_left, pt1_right, pt2_right):

    # Frame preprocessing for deep learning
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    yolo_net.setInput(blob)

    # Getting only output layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = yolo_net.forward(ln)

    # print(len(outputs))
    # for out in outputs:
    #    print(out.shape)

    # Preparing lists for detected bounding boxes, confidences, and class numbers.
    boxes = []
    confidences = []
    classIDs = []

    # Going through all output layers after the feed-forward pass
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # minimum probability to eliminate weak predictions
            if classID > 0.5:
                box = detection[0:4] * np.array([img.shape[1], img.shape[0],
                                                 img.shape[1], img.shape[0]])
                x_center, y_center, box_width, box_height = box
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Implementing non-maximum suppression of given bounding boxes
    # With this technique we exclude some of bounding boxes if their
    # corresponding confidences are low or there is another
    # bounding box for this region with higher confidence
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.)

    region_selected_lane = np.array([[(pt1_left[0], pt1_left[1]), (pt2_left[0], pt2_left[1]), (pt2_right[0], pt2_right[1]), (pt1_right[0], pt1_right[1])]], dtype=np.int32)
    # region_selected_direction = np.array([[(pt1_left[0], pt1_left[1] - 750), (pt2_left[0], pt2_left[1] + 190), (pt2_right[0], pt2_right[1] + 390), (pt1_right[0], pt1_right[1] - 690)]], dtype=np.int32)
    # cv2.line(img, (pt1_left[0], pt1_left[1] - 750), (pt2_left[0], pt2_left[1] + 190), COLOR, THICKNESS, cv2.LINE_AA)
    # cv2.line(img, (pt2_right[0], pt2_right[1] + 390), (pt1_right[0], pt1_right[1] - 690), COLOR, THICKNESS, cv2.LINE_AA)

    # At least one detection should exist
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            intersection_dentro = cv2.pointPolygonTest(region_selected_lane[0], (x + w // 2, y + h // 2),False)
            # intersection_fora = cv2.pointPolygonTest(region_selected_direction[0], (x + w // 2, y + h // 2),False)

            if intersection_dentro >= 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # elif intersection_fora >= 0:
            #    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img


if __name__ == '__main__':

    net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo_confidence_threshold = 0.5

    VideoAquisition()

    success = BuildKernel()
    if not success:
        print('Failed to build OpenCL kernel.')

    while (1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break

        # frame = cv2.resize(frame, (840, 440))

        imageBGRA = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        img_sobel = SobelKernel(imageBGRA)

        imageBGRA = cv2.cvtColor(img_sobel, cv2.COLOR_BGR2BGRA)
        hough_array, hough_array2 = HoughKernel(imageBGRA)
        if not hough_array.any():
            print('Failed to set up OpenCL memory buffers.')
            break

        frame_lines, pt1_left, pt2_left, pt1_right, pt2_right = DrawLines(frame, hough_array, hough_array2)

        frame_cars = detect_cars(frame, net, pt1_left, pt2_left, pt1_right, pt2_right)

        cv2.imshow('frame_dst', frame_cars)
        k = cv2.waitKey(30) & 0xff
