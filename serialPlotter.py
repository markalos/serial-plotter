# -*- coding: utf-8 -*-
# System modules
from Queue import Queue
from threading import Thread
import time
import serial as pyserial
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class CONST(object):
    """const have properties cannot be changed, once it has been defined"""
    def __init__(self, arg):
        super(const, self).__init__()
        self.arg = arg
    def __setattr__(self,name,value):
        if self.__dict__.has_key(name):
            raise self.ConstError, "Can't rebind const(%s)"%name
        self.__dict__[name]=value
    def __delattr__(self, name):
        if self.__dict__.has_key(name):
            raise self.ConstError, "Can't unbind const(%s)"%name
        raise NameError, name

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

def multiarr(x,y):
    length = len(x)
    asum = 0
    for i in xrange(length):
        asum += x[i] * y[i]
    return asum

def butterworth(signal, filtereddata):
    buttercoefoutputflipped = [
    0.069417,
   -0.710299,
    3.293422,
   -8.885405,
   15.355702,
  -17.491016,
   12.765283,
   -5.397104,

  # -5.400162,
  #  12.766616,
  # -17.467434,
  #  15.300848,
  #  -8.828981,
  #   3.261192,
  #  -0.700210,
  #   0.068131,
    ]
    buttercoefinputflipped = [
   0.02064,
   0.00000,
  -0.08257,
   0.00000,
   0.12386,
   0.00000,
  -0.08257,
   0.00000,
   0.02064,

  #    0.02108,
  #  0.00000,
  # -0.08431,
  #  0.00000,
  #  0.12646,
  #  0.00000,
  # -0.08431,
  #  0.00000,
  #  0.02108,
    ]
    nstate = len(buttercoefinputflipped)
    length = len(signal)
    x = nstate
    filtered = np.zeros(length)
    while x < length:
        filtered[x] = multiarr(signal[x - nstate + 1: x + 1], buttercoefinputflipped) - \
        multiarr(filtered[x - nstate + 1: x], buttercoefoutputflipped)
        x += 1
        # print 'x = ', x
    return filtered[-length:]

'''
signal struct [[line1],[line2],...,[line N]], N is 8 in first attempt
'''

def retriveData(fifoqueue, portname):
    # portname = "\\\\.\\PORTB"
    # portname = 'COM7'
    baudrate = 115200
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

    channel8 = 'channel8.txt'
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
                # recheckpos = signal.find(startflag, startpos + 4)
                # if recheckpos - startpos < 28:
                #     startpos = endpos
                #     continue
                # if signal[startpos + 1: startpos + 4] == goodstate:
                #     print '\rgoods state',
                # else :
                #     print '\rbad  state',
                startpos += 4
                for x in xrange(nchannel):
                    # tmp = (ord(signal[startpos]) << 16) + \
                    # (ord(signal[startpos + 1]) << 8) + \
                    # (ord(signal[startpos + 2]))
                    hexdata = signal[startpos:startpos + 3].encode('hex')
                    tmp = int(hexdata, 16)
                    # convert to voltage
                    if (tmp & 0x800000):
                        # tmp |= ~0xffffff
                        tmp = -16777216 + tmp
                    voltage = TOVOLTAGE * tmp
                    # if (x == 7):
                    #     logchannel.write(str(voltage))
                    #     logchannel.write('\n')
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
0.0928650854402458,
-1.31939951937605,
8.63848490798497,
-34.4667916997378,
93.3417483826349,
-180.759904333355,
256.656908970829,
-269.192336747760,
206.949953803772,
-113.692972030114,
42.3485207866428,
-9.59707760695505,
    ]
    buttercoefinputflipped = [
0.000276171799068570,
0.0,
-0.00165703079441142,
0.0,
0.00414257698602855,
0.0,
-0.00552343598137140,
0.0,
0.00414257698602855,
0.0,
-0.00165703079441142,
0.0,
0.000276171799068570
    ]
    nstate = len(buttercoefinputflipped) # filter and order specific
    signalstate = [[0 for i in xrange(framewinsize)] for i in xrange(nchannel)]
    def bandpsss(nth, nnewdata):
        return datacache[nth][-framewinsize:]
        
    def butterworthfilter(nth, nnewdata):
        signalstate[nth][:-nnewdata] = signalstate[nth][nnewdata:]
        index = framewinsize - nnewdata
        bindex = len(datacache[0]) - nnewdata
        while index < framewinsize:
            signalstate[nth][index] = np.dot(buttercoefinputflipped, datacache[nth][bindex - nstate + 1:bindex + 1]) - \
            np.dot(buttercoefoutputflipped, signalstate[nth][index - nstate + 1:index])
            index += 1
        return signalstate[nth]

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
            filtereddata = [[] for i in xrange(nchannel)]
            for i in xrange(nchannel):
                filtereddata[i] = butterworthfilter(i, nnewdata)
            return filtereddata, remainder
        else :
            return [], remainder

    with pyserial.Serial(portname, baudrate, timeout = 0) as mser:
        print 'port description = ',mser
        gooddata = True
        with open('readfromserial.txt', 'wb') as serialfile:
            while gooddata:
                signal = mser.read(1000)
                
                if len(signal):
                    serialfile.write(signal)
                data, presignal = handleData(signal)

                if len(data) > 0:
                    fifoqueue.put(data)
                # time.sleep(2)


def drawData(fifoqueue):
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
        while fifoqueue.qsize() > 1:
            fifoqueue.get()

        if not fifoqueue.empty():
            # assume data pulled out is non empty
            data = fifoqueue.get()
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
                             frames=8, interval=20, blit=True)

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
    portname = '\\\\.\\PORTB'
    usableport = list_serial_ports()
    if len(usableport) == 0:
        print 'there is no usable port, please check your device'
    if len(usableport) == 1:
        portname = usableport[0]
    fifoqueue = Queue(10)
    print 'portname = ', portname
    tgetdata = Thread(target=retriveData, args = (fifoqueue, portname, ))
    tdrawdata = Thread(target=drawData3, args = (fifoqueue,))
    
    # try:
    tgetdata.start()
    tdrawdata.start()
    # except (KeyboardInterrupt, SystemExit):
    #     raise e


if __name__ == '__main__':
    main()