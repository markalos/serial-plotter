# -*- coding: utf-8 -*-
# System modules
from Queue import Queue
from threading import Thread
import time
import serial as pyserial
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

def list_serial_ports(extra=False):
	""" Lists serial port names

		:raises EnvironmentError:
			On unsupported or unknown platforms
		:returns:
			A list of the serial ports available on the system
	"""
	import sys
	if sys.platform.startswith('win'):
		ports = ['COM%s' % (i) for i in range(256)]
		if extra:
			ports += ['\\\\.\\PORTB']
	elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
		# this excludes your current terminal "/dev/tty"
		ports = glob.glob('/dev/tty[A-Za-z]*')
	elif sys.platform.startswith('darwin'):
		ports = glob.glob('/dev/tty.*')
	else:
		raise EnvironmentError('Unsupported platform')

	result = []
	for port in ports:
		try:
			s = pyserial.Serial(port)
			s.close()
			result.append(port)
		except (OSError, pyserial.SerialException):
			pass
	return result


'''
signal struct [[AAC0000F],[line1],[line2],...,[line N]], N is 8 in first attempt
'''

def retriveData(rawdataqueue, portname):
	baudrate = 115200
	# timeout = 0 --- non block reading
	with pyserial.Serial(portname, baudrate, timeout = 0) as mser:
		print 'port description = ',mser
		while True:
			data = mser.read(16)
			if len(data):
				rawdataqueue.put(data)


def handleSignal(figqueue, rawdataqueue):
	bytesize = 8
	datasize = 8 * bytesize
	onsecond = 1
	startflag = chr(0xAA)
	goodstate = chr(0xC0) + chr(0x00) + chr(0x0F)
	startflag += goodstate
	framewinsize = 250
	datainbyte = 28
	signalinbyte = datainbyte - len(startflag)
	nchannel = signalinbyte / 3
	presignal = '' #unhandle signal, cache to next round
	strstoppos = -1
	# in MV
	TOVOLTAGE =  5000.0 / 24.0 / (1 << 24)

	channel8 = 'channel9.txt'
	logchannel = open(channel8, 'w')
	
	# def caculatevoltage(num):
	#     if num & 0x800000:
	#         num 

	'''
	@signal string type, raw byte data
	%data list type
	%remainder string type
	'''
	def decodesignal(signal):
		if len(signal):
			startpos = signal.find(startflag)
			if strstoppos == startpos:
				return [], signal
			endpos = len(signal)
			newdata = [[] for i in xrange(nchannel)]
			while (endpos - startpos) >= datainbyte :
				startpos += 4
				for x in xrange(nchannel):
					hexdata = signal[startpos:startpos + 3].encode('hex')
					tmp = int(hexdata, 16)
					# convert to voltage
					if (tmp & 0x800000):
						tmp |= ~0xffffff
					voltage = TOVOLTAGE * tmp
					newdata[x].append(voltage)
					startpos += 3
				if (startpos >= endpos) :
					return newdata, ''
				if (signal[startpos: startpos + 3] != startflag):
					break
			if (len(newdata[0]) == 0):
				newdata = []
			return newdata, signal[startpos:]

		else :
			return [], signal
	# cache 3 sec signal data
	cachesizeperchannel = framewinsize * 3
	dropthreshold = cachesizeperchannel / 2
	datacache = [[0 for i in xrange(framewinsize)] for i in xrange(nchannel)]

	
	buttercoefoutputflipped = [
0.119946687018252,
-1.67693670075963,
10.7937050673710,
-42.2961849734859,
112.384259923576,
-213.311658428113,
296.551487755547,
-304.231194211049,
228.550302435025,
-122.588320952459,
44.5492914014245,
-9.84469800409247
	]
	buttercoefinputflipped = [
0.000156275168096088,
0,
-0.000937651008576530,
0,
0.00234412752144133,
0,
-0.00312550336192177,
0,
0.00234412752144133,
0,
-0.000937651008576530,
0,
0.000156275168096088
	]
	nstate = len(buttercoefinputflipped) # filter and order specific
	signalstate = [[0 for i in xrange(framewinsize)] for i in xrange(nchannel)]
	def bandpsss(nth, nnewdata):
		return datacache[nth][-framewinsize:]
		
	def butterworthfilter(nnewdata):
		for nth in xrange(nchannel):
			signalstate[nth][:-nnewdata] = signalstate[nth][nnewdata:]
			index = framewinsize - nnewdata
			bindex = len(datacache[0]) - nnewdata
			while index < framewinsize:
				signalstate[nth][index] = np.dot(buttercoefinputflipped, datacache[nth][bindex - nstate + 1:bindex + 1]) - \
				np.dot(buttercoefoutputflipped, signalstate[nth][index - nstate + 1:index])
				index += 1

	peakthreshold = 0.3

	def handlePeak(signal, nnewdata):
		index = len(signal) - nnewdata
		while index < len(signal) :
			# if (abs(signal[index - 1]) > peakthreshold) and (abs(signal[index] -signal[index - 1]) > abs(signal[index -1]) * peakthreshold):
			if (abs(signal[index] - 8) > 1):
				signal[index] = signal[index - 1]# + (signal[index] - signal[index - 1]) * peakthreshold
			index += 1
		return signal

	def handleData(signal):
		if type(presignal) != str:
			print '\rexception history = ', presignal,
		signal = presignal + signal
		data, remainder = decodesignal(signal)

		if len(data):
			nnewdata = len(data[0])
			for i in xrange(nchannel):
				datacache[i] += data[i]
				# datacache[i] = handlePeak(datacache[i], nnewdata)
			# drop out-of-date data
			if len(datacache[0]) > cachesizeperchannel:
				for i in xrange(nchannel):
					datacache[i] = datacache[i][dropthreshold:]
			# filter data
			butterworthfilter(nnewdata)
			for x in signalstate[7][-nnewdata:]:
				logchannel.write(str(x) + '\n')
			return signalstate[:], remainder
		else :
			return [], remainder

	with open('writeserialdata.txt', 'wb') as serialfile:
		while True:
			signal = rawdataqueue.get()
			if len(signal):
				serialfile.write(signal)
			data, presignal = handleData(signal)
			if len(data) > 0:
				figqueue.put(data)


def drawData(figqueue):
	framewinsize = 250
	nchannel = 8
	onsecond = 1

	fig,ax = plt.subplots()
	ax.set_xlim(0, onsecond)
	ax.set_ylim(-1, 1)
	xcor = np.linspace(0, onsecond, framewinsize)
	ycor = np.zeros(framewinsize)

	channels = [plt.plot([], [])[0] for _ in xrange(nchannel)]

	def init():
		for line in channels:
			line.set_data(xcor,ycor)
		return channels

	def animate(time):
		while figqueue.qsize() > 1:
			figqueue.get()

		if not figqueue.empty():
			# assume data pulled out is non empty
			data = figqueue.get()
			# yupper = np.amax(data)
			# ylow = np.amin(data)
			for j, line in enumerate(channels):
				
				line.set_ydata(data[j])
			# ax.set_ylim(ylow, yupper)
		return channels
	ax.relim()
	ax.autoscale_view()
	anim = animation.FuncAnimation(fig, animate, init_func=init,
							 frames=nchannel, interval=20, blit=True)

	plt.show()


def drawData3(fifoqueue):
	framewinsize = 250
	epsilon = 0.000001
	onsecond = 1
	nfig = 1
	fig,axarr = plt.subplots(nfig, sharex = True, sharey = True)
	axarr.set_xlim(0, onsecond)
	axarr.set_ylim(-1, 1)
	xcor = np.linspace(0, onsecond, framewinsize)
	ycor = np.zeros(framewinsize)

	channels = [axarr.plot([], [])[0] for i in xrange(1)]

	def init():
		for line in channels:
			line.set_data(xcor,ycor)
		return channels

	def animate(time):
		while fifoqueue.qsize() > 1:
			fifoqueue.get()

		if not fifoqueue.empty():
			# assume data pulled out is non empty
			data = fifoqueue.get()
			# yupper = np.amax(data)
			# ylow = np.amin(data)
			upper = max(data[7])
			low = min(data[7])
			# if j == 7:
			#     print 'limit', low ,upper
			if abs(upper - low) < epsilon:
				upper = low + 0.01
				low = low - 0.01
			axarr.set_ylim(low, upper)
			channels[0].set_ydata(data[7])
			# ax.set_ylim(ylow, yupper)
		return channels

	anim = animation.FuncAnimation(fig, animate, init_func=init,
							 frames=8, interval=100, blit=True)

	plt.show()

def drawData2(fifoqueue):
	framewinsize = 250
	epsilon = 0.000001
	nchannel = 8
	onsecond = 1
	nfig = 8
	fig,axarr = plt.subplots(nfig, sharex = True, sharey = True)
	for i in xrange(nchannel):
		axarr[i % nfig].set_xlim(0, onsecond)
		axarr[i % nfig].set_ylim(-0.1, 0.1)
	xcor = np.linspace(0, onsecond, framewinsize)
	ycor = np.zeros(framewinsize)

	channels = [axarr[i % nfig].plot([], [])[0] for i in xrange(nchannel)]

	def init():
		for line in channels:
			line.set_data(xcor,ycor)
		return channels

	def animate(time):
		while fifoqueue.qsize() > 1:
			fifoqueue.get()

		if not fifoqueue.empty():
			# assume data pulled out is non empty
			data = fifoqueue.get()
			# yupper = np.amax(data)
			# ylow = np.amin(data)
			for j, line in enumerate(channels):
				upper = max(data[j])
				low = min(data[j])
				# if j == 7:
				#     print 'limit', low ,upper
				if abs(upper - low) < epsilon:
					upper = low + 0.01
					low = low - 0.01
				axarr[j].set_ylim(low, upper)
				line.set_ydata(data[j])
			# ax.set_ylim(ylow, yupper)
		return channels

	anim = animation.FuncAnimation(fig, animate, init_func=init,
							 frames=nchannel, interval=20, blit=True)

	plt.show()
	

def generatedata(fifoqueue, portname):
	nchannel = 8
	framewinsize = 250
	while True:
		data = [np.random.rand(framewinsize) for i in xrange(nchannel)]
		fifoqueue.put(data)

def main():
	import sys
	portname = '\\\\.\\PORTB'
	if len(sys.argv) > 1:
		usableport = list_serial_ports(True)
	else:
		usableport = list_serial_ports(False)
	if len(usableport) == 0:
		print 'there is no usable port!!!!!!!!'
	if len(usableport) == 1:
		portname = usableport[0]
	rawdataqueue = Queue(10)
	figdataqueue = Queue()
	print 'portname = ', portname
	tgetdata = Thread(target=retriveData, args = (rawdataqueue, portname, ))
	tdrawdata = Thread(target=drawData3, args = (figdataqueue,))
	thandledata = Thread(target = handleSignal, args= (figdataqueue, rawdataqueue, ))
	# try:
	tgetdata.start()
	thandledata.start()
	tdrawdata.start()
	# except (KeyboardInterrupt, SystemExit):
	#     raise e


if __name__ == '__main__':
	main()