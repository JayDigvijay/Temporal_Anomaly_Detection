mport csv
import subprocess
import psutil

def checkIfProcessRunning(processName):
    
    '''
    Check if there is any running process that contains the given name processName.
    '''
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

while (checkIfProcessRunning('python3') and i < 100):
    CPUPerc = subprocess.run(" top -n 1| awk '/python3/ {print $10}'", shell = True, stdout = subprocess.PIPE)
	data2 = CPUPerc.stdout.decode('utf-8')
	data2 = data2.split("\n")
	data2 = data2[:-1]
    CPU = float(data2)
    print("CPU Percentage ", i, " is ", "{0:.2f}".format(CPU), "%")
