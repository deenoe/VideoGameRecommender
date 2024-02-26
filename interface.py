# file of base interface
# no need

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

def import_csv():
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])

    if file_path:
        try:
            # perform import logic here
            # for demonstration purposes, let's assume import is successful
            messagebox.showinfo("Import Successful", "CSV file imported successfully!", icon='info')
        except Exception as e:
            # handle import failure
            messagebox.showerror("Import Failed", f"Failed to import CSV file.\nError: {str(e)}", icon='error')

def run_function():
    # placeholder for the function to be called when the "Run" button is clicked
    messagebox.showinfo("Run Button", "Run function called!", icon='info')

# create the main Tkinter window
root = tk.Tk()
root.title("KNN Cluster Video Game Recommender")
root.geometry("800x600")
root.configure(bg="purple")

# load and display an image (replace "example_image.png" with your image file)
image_path = "videogame_reccomender.png"
image = Image.open(image_path)
photo = ImageTk.PhotoImage(image)

image_label = tk.Label(root, image=photo, bg="purple")
image_label.image = photo  # keep a reference to prevent garbage collection
image_label.pack(pady=10)

# add a white title label
title_label = tk.Label(root, text="KNN Cluster Video Game Recommender", font=("Helvetica", 16, "bold"), bg="purple", fg="white")
title_label.pack(pady=10)

# add a title above the Select CSV File section
file_title_label = tk.Label(root, text="Video Game Recommender", font=("Helvetica", 12, "bold"), bg="purple", fg="white")
file_title_label.pack(pady=5)

# add a text field for file path
file_path_entry = tk.Entry(root, width=40)
file_path_entry.pack(pady=10)

# add a button to open the file dialog
select_file_button = tk.Button(root, text="Select CSV File", command=import_csv)
select_file_button.pack(pady=10)

# add a "Run" button
run_button = tk.Button(root, text="Run", command=run_function)
run_button.pack(pady=10)

# add a label to display import status
status_label = tk.Label(root, text="", fg="green", bg="purple")
status_label.pack(pady=10)

# add a Done button at the top right
done_button = tk.Button(root, text="Done", command=root.destroy)
done_button.pack(side=tk.TOP, anchor=tk.NE, padx=10, pady=5)

# start the Tkinter event loop
root.mainloop()
