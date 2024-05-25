import cv2
import os
import tkinter as tk
from tkinter import messagebox

dataPath = r'C:\Users\SOPORTE\OmesTutorials2020\OmesTutorials2020\6 RECONOCIMIENTO FACIAL\data' # Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Inicialización de la captura de video desde la cámara
# cap = cv2.VideoCapture('Video.mp4')  # Inicialización de la captura desde un archivo de video

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para verificar el login
def check_login():
    username = username_entry.get()
    password = password_entry.get()

    if username == "timoteo" and password == "timjus10":
        messagebox.showinfo("Login Exitoso", "Bienvenido, " + username)
        login_window.destroy()  # Cerrar la ventana de login
    else:
        messagebox.showerror("Error de Login", "Nombre de usuario o contraseña incorrectos")

# Crear ventana de login
login_window = tk.Tk()
login_window.title("Login")
login_window.geometry("300x200")  # Tamaño de la ventana

# Colores
bg_color = "#2c3e50"  # Color de fondo
fg_color = "#ffffff"  # Color de texto
button_color = "#3498db"  # Color del botón

login_window.configure(bg=bg_color)

# Campos de entrada para usuario y contraseña
username_label = tk.Label(login_window, text="Nombre de Usuario:", bg=bg_color, fg=fg_color)
username_label.pack()
username_entry = tk.Entry(login_window)
username_entry.pack()

password_label = tk.Label(login_window, text="Contraseña:", bg=bg_color, fg=fg_color)
password_label.pack()
password_entry = tk.Entry(login_window, show="*")
password_entry.pack()

# Botón de login
login_button = tk.Button(login_window, text="Iniciar Sesión", command=check_login, bg=button_color, fg=fg_color)
login_button.pack(pady=10)

login_window.mainloop()  # Mostrar la ventana de login

# Loop principal
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if result[1] < 70:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
