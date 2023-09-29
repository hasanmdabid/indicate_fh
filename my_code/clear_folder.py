import os
import shutil

def clear_folder(folder_path):
    try:
        # Ensure the folder path exists
        if not os.path.exists(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return

        # Remove all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        # Remove all subdirectories and their contents
        for dir_name in os.listdir(folder_path):
            dir_path = os.path.join(folder_path, dir_name)
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)

        print(f"Contents of '{folder_path}' have been cleared.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

