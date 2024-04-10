#This will sort through your files and organize them into folders named "Patientx"
#It is important the images are in folders named the same for the video code
import os
import shutil


# Directory will change based on the images you want organized. Make sure you change the path for where your images are located
directory = ("Foo")


def sort_images(directory):
    # Create folders for patients if they don't exist
    for i in range(1, 5):  # Assuming patients are numbered from 1 to 4
        patient_folder = os.path.join(directory, f"patient{i}")
        if not os.path.exists(patient_folder):
            os.makedirs(patient_folder)

    # Loop through files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # You can change the extension as per your images
            # Extract patient number from filename
            patient_num = filename.split("_")[2].split(".")[0]
            try:
                patient_num = int(patient_num)
                if 1 <= patient_num <= 4:  # Assuming patients are numbered from 1 to 4
                    # Construct destination folder path
                    destination_folder = os.path.join(directory, f"patient{patient_num}")
                    # Move the file to the appropriate folder
                    shutil.move(os.path.join(directory, filename), destination_folder)
                    print(f"Moved {filename} to {destination_folder}")
                else:
                    print(f"Illegal patient number in filename: {filename}")
            except ValueError:
                print(f"Unable to extract patient number from filename: {filename}")
        else:
            print(f"Skipping non-image file: {filename}")



sort_images(directory)
