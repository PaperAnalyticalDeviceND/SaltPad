#!/usr/bin/env python
from Tkinter import *
import tkFileDialog, tkMessageBox
import datetime, os
import sys
import subprocess
import fileinput

####################################################################################################
# Chris Sweet. Center for Research Computing. Notre Dame.
# 05/28/2014
# GUI to wrap pad.py analysis code and the .sh scripts
####################################################################################################
class App:

    filename = ""

    def __init__(self, master):
        self.master = master
        self.start()

    def start(self):
        self.master.title("saltPAD Analysis Tool v0.9")
        self.master.configure(borderwidth = 10)#bg = 'grey')
        self.master.resizable(FALSE,FALSE)

        self.now = datetime.datetime.now()
        label01 = "Browse for Image file name, then select EXECUTE."
        Label(self.master, text=label01).grid(row=0, column=0, sticky=W)

        self.fileloc = Entry(self.master)
        self.fileloc["width"] = 60
        self.fileloc.insert(0,"Input file/folder")
        self.fileloc.focus_set()
        self.fileloc.grid(row=1, column=0)

        self.open_file = Button(self.master, text="Browse...", command=self.browse_file, width=8)
        self.open_file.grid(row=1, column=1)

        Label(self.master, text="Processing type:").grid(row=3, column=0, sticky=W)
        ADJUSTMENTS = [
            ("Single image", "1"),
            ("All images in folder", "2")
        ]

        self.adj = StringVar()
        self.adj.set("1") #Initialize

        i = 4
        for text, mode in ADJUSTMENTS:
            self.adjustment = Radiobutton(self.master, text=text, variable=self.adj, value=mode)
            self.adjustment.grid(row=i, column=0, sticky=W)
            i += 1

        self.loccsv = IntVar()
        self.loccsv.set(0) #Initialize

        self.check = Checkbutton(self.master, text="Local CSV", variable=self.loccsv)
        self.check.grid(row=6, column=0, sticky=W)

        #set row for next widget
        i = 7

        #do we need dropdown box for calibration data?
        self.optvar = StringVar(self.master)
        self.optvar.set("") # initial value
        sets = []
        currentset = ""
        try:
            f = open("calibration.csv", 'rU')
            if f:
                for line in f:
                    if "#" in line:
                        continue
                    elif "calibration" in line:
                        split = line.split(",")
                        sets.append(split[1])
                    elif "selected" in line:
                        split = line.split(",")
                        currentset = split[1]
            f.close()
        except:
            print "Calibration file error."

        #any sets defined?
        if len(sets) > 0:
            Label(self.master, text="Select Calibration set.").grid(row=i, column=0, sticky=W)
            i += 1
            if currentset != "":
                self.optvar.set(currentset) # initial value
            else:
                self.optvar.set(sets[0]) # initial value
            self.option = OptionMenu(self.master, self.optvar, *sets)#"one", "two", "three", "four")
            self.option.grid(row=i, column=0, sticky=W)
            i += 1

        #text box
        self.text = Text(self.master, height=25, width=70, borderwidth=3, bg="Snow2")
        self.text.grid(row=i, column=0, sticky=W)
        self.text.insert(END, "Analysis output,\n")
        i += 1

        #bottom label
        Label(self.master, text="University of Notre Dame, Notre Dame, IN 46556, USA.").grid(row=i, column=0, sticky=W)

        self.submit = Button(self.master, text="EXECUTE", command=self.start_processing, fg="red", width=8)
        self.submit.grid(row=3, column=1, sticky=E)

        self.submit = Button(self.master, text="Save", command=self.save, width=8)
        self.submit.grid(row=4, column=1, sticky=E)

        self.submit = Button(self.master, text="Exit", command=self.exit, width=8)
        self.submit.grid(row=5, column=1, sticky=E)

    def save(self):
        tmpfilename = tkFileDialog.asksaveasfilename(title="Save csv file...")
        if tmpfilename:
            myFile = open(tmpfilename,'w')
            myFile.write(self.text.get(1.0, END))
            myFile.close()


    def exit(self):
        self.master.destroy()

    def browse_file(self):
        if self.adj.get() == "1":
            self.filename = tkFileDialog.askopenfilename(title="Open image file...")
        else:
            self.filename = tkFileDialog.askdirectory(title="Open image folder...")
        self.fileloc.delete(0, END)
        self.fileloc.insert(0,self.filename )#set the location to fileloc var

    def start_processing(self):
        try:
            print "Subprocess", self.filename
            #get cal lab data if available
            callab = self.optvar.get()
            if callab != "":
                #switch selected value
                print "Calibration set data exists",callab
                for line in fileinput.input("calibration.csv", inplace=True, mode="U"):
                    if "selected," in line:
                        print "selected,"+callab+","
                    else:
                        print line,
            self.text.insert(END, "Using calibration set: "+callab+"\n")
            inp = []
            if self.adj.get() == "1":
                if self.loccsv.get() == 0:
                    inp = ["python", "pad.py", "-o", "auto", "-t", "template2.png", "-c", "calibration.csv", self.filename]
                else:
                    inp = ["python", "pad.py", "-o", "local.csv", "-t", "template2.png", "-c", "calibration.csv", self.filename]
            else:
                if self.loccsv.get() == 0:
                    inp = ["./readsaltpad.sh", self.filename]
                else:
                    inp = ["./readsaltpadLocalcsv.sh", self.filename]
            p = subprocess.Popen(inp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (stdout, stderr) = p.communicate()
            dataLines = stdout.split("\n")
            #print to stdio and sift to text box
            addToResults = False
            for i in range(0,len(dataLines)):
                print dataLines[i]
                #start
                if 'File name' in dataLines[i]:
                    addToResults = True
                if addToResults:
                    self.text.insert(END, dataLines[i]+"\n")
                #end, was '3, 2,'
                if 'End of analysis.' in dataLines[i]:
                    addToResults = False

            self.text.insert(END, '-----------------------------------------------------------------,\n')
            tkMessageBox.showinfo("Results", "Finished analysis")
        except:
            tkMessageBox.showerror("Unexpected Error", sys.exc_info())

#set to my folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#instantiate
root = Tk()
app = App(root)
root.mainloop()
