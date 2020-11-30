
import csv
import subprocess
import psutil
import time

f1 = open('CPU_Usage.csv', 'w', newline = '')
f1.truncate()
CPUWriter = csv.writer(f1)
f2 = open('Memory_Usage.csv', 'w', newline = '')
f2.truncate()
MemWriter = csv.writer(f2)


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
i = 0
while (checkIfProcessRunning('python3')):
	CPUPerc = subprocess.run(" top -n 1| awk '/python3/ {print $9}'", shell = True, stdout = subprocess.PIPE)
	data2 = CPUPerc.stdout.decode('utf-8')
	data2 = data2.split("\n")
	data2 = data2[:-1]
	#print(data2)
	CPU = 0
	time.sleep(2)
	i += 1
	for x in data2:
		CPU += float(x)
	MemPerc = subprocess.run(" top -n 1| awk '/python3/ {print $10}'", shell = True, stdout = subprocess.PIPE)
	data2 = MemPerc.stdout.decode('utf-8')
	data2 = data2.split("\n")
	data2 = data2[:-1]
	#print(data2)
	Mem = 0
	for x in data2:
		Mem += float(x)
	
	CPUWriter.writerow([int(CPU)])
	MemWriter.writerow([int(Mem)])
	print("CPU Percentage ", i, " is ", "{0:.2f}".format(CPU), "%")
	print("Memory Percentage ", i, " is ", "{0:.2f}".format(Mem), "%")
	f1.flush()
	f2.flush()
	if CPU < 1.00 and Mem < 1.00:
		print("Process Finished")
		f1.close()
		f2.close()
		break
