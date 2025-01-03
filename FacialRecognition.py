import os
import cv2
import dlib
import numpy

def findEuclideanDistance(source_representation, test_representation):
    """
    Calculates representation value
    """
    euclidean_distance = source_representation - test_representation
    euclidean_distance = numpy.sum(numpy.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = numpy.sqrt(euclidean_distance)
    return euclidean_distance

#extra values
similarity_threshold = 0.5

#grab face detector
print("Loading facial recognition...")
face_detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
model = dlib.face_recognition_model_v1("data/dlib_face_recognition_resnet_model_v1.dat")

#grab people data
print("Loading data...")
people = {}

dir_list = os.listdir("People")

for dir in dir_list:
    print(f":: Loading {dir} data...")
    level_list = os.listdir(os.path.join("People", dir))
    levels = {}

    for level in level_list:
        people_list = os.listdir(f"People\\{dir}\\{level}")
        face_representations = []

        for person in people_list:
            image = dlib.load_rgb_image(f"People\\{dir}\\{level}\\{person}")
            image_faces = face_detector(image, 1)

            for image_face in image_faces:
                image_shape = sp(image, image_face)
                aligned_image = dlib.get_face_chip(image, image_shape)
                image_representation = model.compute_face_descriptor(aligned_image)

                face_representations.append(numpy.array(image_representation))

        levels[level] = face_representations

    people[dir] = levels

#grab camera
print("Loading camera...")
camera = cv2.VideoCapture(0)

#open new window
print("Loading display...")
win = dlib.image_window()

while True:
    #grab video frame from camera as rgb image
    _, bgr_frame = camera.read()
    frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    #detect faces
    faces = face_detector(frame, 1)

    #remove overlays from window
    win.clear_overlay()

    #run through faces
    for face in faces:
        #grab face shape data
        face_shape = sp(frame, face)
        aligned_face = dlib.get_face_chip(frame, face_shape)
        face_representation = model.compute_face_descriptor(aligned_face)

        lowest_distance = 1
        level = ""
        person = ""
        #find each representation value for each person
        for level_name in people:
            #scan through each person
            for person_name in people[level_name]:
                distance = 0
                index = 0

                #compare each face
                for representation in people[level_name][person_name]:
                    index = index + 1
                    distance = distance + findEuclideanDistance(face_representation, representation)

                #find average representation value of each person
                if index > 0:
                    average = distance / index

                    if average < lowest_distance:
                        lowest_distance = average
                        level = level_name
                        person = person_name

        #display level and person
        text_display = "NOT RECOGNIZED"
        if lowest_distance <= similarity_threshold:
            text_display = f"{person} :: {level}"
        text_width, text_height = cv2.getTextSize(text_display, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.putText(frame, text_display, (int(((face.left() + face.right()) / 2) - (text_width / 2)), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #display shape data in window
        win.add_overlay(face_shape)

    #set window to frame
    win.set_image(frame)
