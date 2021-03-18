import jetson.inference
import jetson.utils

import json
import time
import uuid
import threading

from MotionPrediction import associate_points
from myInflux import MyInflux
from Person import Person

import cv2
import numpy as np

import signal
import sys

def sigint_handler(signal, frame):
    print('__Interrupted__')
    global _stop_polifemo_
    _stop_polifemo_ = True
signal.signal(signal.SIGINT, sigint_handler)

people_in, people_out = 0,0
going_in, going_out = 0,0
cameraID = uuid.getnode()

dispW=1280
dispH=720
flip=2

_stop_polifemo_ = False

lock = threading.Lock()


#funzione usata per inviare i dati al db
#eseguita in un thread separato, invia i dati ad intervalli di 5 secondi
def update_db(db_client, cameraID):

    global people_in, people_out
    global going_in, going_out

    while not _stop_polifemo_:
        
        #crossed: persone che hanno attraversato la linea immaginaria
        db_client.write_crossed(going_in, going_out, cameraID)

        #revealed: numero di persone individuate da un lato e dall' altro della linea
        db_client.write_revealed(people_in, people_out, cameraID)

        with lock:
            #dopo aver scritto le info su db, azzero i valori
            going_in, going_out = 0,0

        #attendo 5 secondi tra una scrittura e l'altra, per non caricare troppo il server del db
        time.sleep(5)

#funzione usata per contare le persone che ci sono in un determinato frame
# e per determinare in quanti hanno attraversato la linea
def count_people(people):

    global people_in, people_out
    global going_in, going_out

    with lock:
            people_in, people_out = 0,0

    for p in people:

        if p.position == 'IN':
            if p.crossed:
                going_in = going_in + 1
            people_in = people_in + 1

        elif p.position == 'OUT':
            if p.crossed:
                going_out = going_out + 1
            people_out = people_out + 1

#funzione usata per stabilire se un punto si trovi a destra o a sinistra di una linea obliqua
# 'a' e 'b' sono i punti della linea, 'c' è il punto da confrontare per stabilirne la posizione
def isLeft(a, b, c):
    return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0 

#funzione che cattura il frame del video e lo elabora con il modello di object detection
def detect(camera, detectnet, camera_config):

    people = []

    #height, width = camera.GetHeight(), camera.GetWidth()
    height, width = dispH, dispW
    point1 = point01 = (0,0)
    point2 = point02 = (width,height)
    line = 0.5

    # leggo informazioni sulla linea dal file di configurazione
    line_position = camera_config.get("line_position") if camera_config.get("line_position") else 0.5
    line_orientation = camera_config.get("line_orientation") if camera_config.get("line_orientation") else 'vertical'
    
    if line_orientation == 'vertical':
        line = width*line_position
    elif line_orientation == 'horizzontal':
        line = height*line_position
    elif line_position.get("point1") and line_position.get("point2"): #se ho definito due punti appartenenti alla linea obliqua desiderata
        point1 = (int(line_position.get("point1").get("x")), int(line_position.get("point1").get("y"))) if line_position.get("point1").get("x") is not None and line_position.get("point1").get("y") is not None else point01
        point2 = (int(line_position.get("point2").get("x")), int(line_position.get("point2").get("y"))) if line_position.get("point2").get("x") is not None and line_position.get("point2").get("y") is not None else point02
    position_in = camera_config.get("position_in") if camera_config.get("position_in") else 'right'

    print(
        "CONFIGURATION DATA:"
        "\n\tposition_in: {}"
        "\n\tline_orientation: {}"
        "\n\tpoint {}"
        .format(
            position_in,
            line_orientation,
            (point1, point2) if line_orientation == 'oblique' else line
        )
    )
    while not _stop_polifemo_:

        t0 = time.time()

        #img = camera.Capture()

        hasFrame, frame = camera.read()

        if hasFrame:

            #cv2 legge le immagini con un formato di colori BGR
            # mentre il modello di object detection se le aspetta in formato RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).astype(np.float32)
            img = jetson.utils.cudaFromNumpy(img)
            detections = detectnet.Detect(img, frame.shape[1], frame.shape[0])

            detected_peolple = []

            for detected_obj in detections:
                if net.GetClassDesc(detected_obj.ClassID) == 'person':

                    center = detected_obj.Center

                    #calcolo la posizine del punto rispetto alla linea
                    # in base a come questa sia stata definita nel file di configurazione
                    position = ''
                    if line_orientation == 'vertical':
                        if center[0] < line and position_in == 'left':
                            position = 'IN'
                        elif center[0] > line and position_in == 'right':
                            position = 'IN'
                        else:
                            position = 'OUT'
                    elif line_orientation == 'horizzontal':
                        if center[1] < line and position_in == 'top':
                            position = 'IN'
                        elif center[1] > line and position_in == 'bottom':
                            position = 'IN'
                        else:
                            position = 'OUT'
                    else:
                        if isLeft(point1, point2, center) and position_in == 'left':
                            position = 'IN'
                        elif not isLeft(point1, point2, center) and position_in == 'right':
                            position = 'IN'
                        else:
                            position = 'OUT'

                    #creo l'oggetto persona e lo aggiungo ad una lista
                    detected_peolple.append(Person(center,position))

                    #disegno le bounding boxes sul frame
                    #utile solo in fase di debug
                    #cv2.rectangle(frame, (int(detected_obj.Left), int(detected_obj.Top)), (int(detected_obj.Right), int(detected_obj.Bottom)), (0, 255, 0), 1)

            if len(people) > 0 and len(detected_peolple) > 0: #se ho persone in memoria a cui associare le persone appena rilevate 
                people = associate_points(people, detected_peolple, width, height)
            else: #altrimenti aggiungo in memoria le persone appena rilevate, e provo ad associarle al prossimo ciclo
                people = detected_peolple

            count_people(people)

            #disegno la linea di separazione sul frame
            #utile solo in fase di debug
            #cv2.line(frame, point1, point2, (255, 0, 0), 1)

            #mostro i frame con le informazini grafiche utili (bounding boxes e linea)
            # utile solo in fase di debug
            #cv2.imshow('detectNet',frame)
            #cv2.moveWindow('detectNet', 0, 35)
            #if cv2.waitKey(1)==ord('q'):
                #global _stop_polifemo_
                #_stop_polifemo_ = True
                #break
            
            #print("PEOPLE ON SCREEN: {} \tFPS: {}".format(len(people),1/(time.time() - t0)))
            time.sleep(0.1)

if __name__ == '__main__':
    
    #leggo il file di configurazione
    with open('./configuration.json') as f:
        configuration_data = json.load(f)

        camera_config = configuration_data.get("cameras")
        influx_config = configuration_data.get("databases")
        args = configuration_data.get("arguments")

        #creo la connessione col db
        db_client = MyInflux(influx_config.get("hostname"),influx_config.get("database_name"),influx_config.get("port"))

        #carico il modello di object detection
        net = jetson.inference.detectNet(args.get("detectNet"), threshold=args.get("threshold"))
        
        #camera = jetson.utils.videoSource(camera_config.get("source"), argv=['--input-width=1920', '--input-height=1080', '--input-rate=15'])      # 3264 x 2464 FPS 21; 3264 x 1848 FPS 28; 1920 x 1080 FPS 30
        
        #comment These next Two Line for Jetson utils
        #apro lo streaming video della raspberry pi camera collegata al jetson
        camSet='nvarguscamerasrc wbmode=2 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 !  video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=2 brightness=-.25 saturation=1.3 ! appsink drop=true'
        camera= cv2.VideoCapture(camSet)

        #eseguo in thread separati la scrittura delle informazioni su db e l'elaborazione dei frame video
        detection = threading.Thread(target=detect, args=[camera, net, camera_config])
        database_writing = threading.Thread(target=update_db, args=[db_client, camera_config.get("name")])

        detection.start()
        database_writing.start()
        detection.join()
        database_writing.join()
        #quando viene interrotto il thread di elaborazione delle immagini, chiudo la connessione col db
        db_client.client.close()

        print('closed db connection..')
        #detect(camera, net, camera_config, db_client, display)
