import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def upload_files():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()       
    drive = GoogleDrive(gauth)

    folder_id = 'your_folder_id_here'

    for file in os.listdir('csv'):
        if file.endswith('.csv'):
            file_path = os.path.join('csv', file)
            gfile = drive.CreateFile({'parents': [{'id': folder_id}], 'title': file})
            gfile.SetContentFile(file_path)
            gfile.Upload()

if __name__ == "__main__":
    upload_files()
