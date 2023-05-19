import random
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import numpy.linalg as la

g = 9.81


class Foll:
    def __init__(self, ):
        self.g = g

    def euler(self, f, horn, hornhradi, fjoldiskrefa, lengd, dempunarstuðull=0):
        skreflengd = lengd / fjoldiskrefa
        skref = 0
        hornaxis = []
        hornhradiaxis = []

        hornaxis.append(horn)
        hornhradiaxis.append(hornhradi)

        if not (0 <= dempunarstuðull <= 10):
            raise ValueError("Dempunarstuðull þarf að vera frá 0 til 10")

        for i in range(0, fjoldiskrefa):
            skref = skref + skreflengd
            hornaxis.append(hornaxis[i] + skreflengd * hornhradiaxis[i])

            if hornhradiaxis[i] != 0:
                dempun = -1 * hornhradiaxis[i] * dempunarstuðull
            else:
                dempun = 0

            hornhradiaxis.append(hornhradiaxis[i] + skreflengd * f(hornaxis[i]) + dempun)
        return hornaxis

    def RKmethod(self, f, horn, hornhradi, fjoldiskrefa, lengd, dempunarstuðull=0):
        skreflengd = lengd / fjoldiskrefa  # h = skreflengd
        skref = 0  # skref = t
        hornaxis = []
        hornhradiaxis = []

        hornaxis.append(horn)
        hornhradiaxis.append(hornhradi)

        if not (0 <= dempunarstuðull <= 10):
            raise "Dempunarstuðull þarf að vera frá 0 til 10"

        for i in range(0, fjoldiskrefa):
            skref = skref + skreflengd
            s1 = f(hornaxis[i])
            s2 = f(hornaxis[i] + skreflengd * (s1 / 2))
            s3 = f(hornaxis[i] + skreflengd * (s2 / 2))
            s4 = f(hornaxis[i] + skreflengd * s3)

            w = hornhradiaxis[i] + (s1 + s2 * 2 + s3 * 2 + s4) / 6 * skreflengd

            if w != 0:
                dempun = -1 * w * dempunarstuðull
            else:
                dempun = 0

            hornhradiaxis.append(w + dempun)
            hornaxis.append(hornaxis[i] + skreflengd * w)

        return hornaxis

    def RKmethod2(self, f1, f2, horn1, horn2, hornhradi1, hornhradi2, fjoldiskrefa, lengd, dempunarstuðull=0):
        skreflengd = lengd / fjoldiskrefa  # h = skreflengd
        skref = 0  # skref = t
        axis = np.zeros((fjoldiskrefa + 1, 4))
        axis[0] = np.array([[horn1, horn2, hornhradi1, hornhradi2]])

        if not (0 <= dempunarstuðull <= 10):
            raise ValueError("Dempunarstuðull þarf að vera frá 0 til 10")

        def f(y):
            th1, th2, thp1, thp2 = y
            if thp1 != 0:
                dempun = -1 * thp1 * dempunarstuðull
            else:
                dempun = 0
            if thp2 != 0:
                dempun2 = -1 * thp2 * dempunarstuðull
            else:
                dempun2 = 0

            return np.array([thp1, thp2, f1(*y) + dempun, f2(*y) + dempun2])

        for i in range(0, fjoldiskrefa):
            skref = skref + skreflengd

            s1 = f(axis[i])
            s2 = f((axis[i] + skreflengd * s1 / 2))
            s3 = f((axis[i] + skreflengd * s2 / 2))
            s4 = f((axis[i] + skreflengd * s3))

            axis[i + 1] = axis[i] + (s1 + s2 * 2 + s3 * 2 + s4) / 6 * skreflengd
        return axis

    def RKmethod3(self, triplediple, horn1, horn2, horn3, hornhradi1, hornhradi2, hornhradi3, fjoldiskrefa, lengd,
                  dempunarstuðull=0):
        skreflengd = lengd / fjoldiskrefa  # h = skreflengd
        skref = 0  # skref = t
        axis = np.zeros((fjoldiskrefa + 1, 6))
        axis[0] = np.array([[horn1, horn2, horn3, hornhradi1, hornhradi2, hornhradi3]])

        if not (0 <= dempunarstuðull <= 10):
            raise "Dempunarstuðull þarf að vera frá 0 til 10"

        def f(y):
            th1, th2, th3, thp1, thp2, thp3 = y

            dempun = -1 * thp1 * dempunarstuðull
            dempun2 = -1 * thp2 * dempunarstuðull
            dempun3 = -1 * thp3 * dempunarstuðull

            svar = triplediple(*y)

            return np.array([thp1, thp2, thp3, *svar[0] + dempun, *svar[1] + dempun2, *svar[2] + dempun3])

        for i in range(0, fjoldiskrefa):
            skref = skref + skreflengd

            s1 = f(axis[i])
            s2 = f((axis[i] + skreflengd * s1 / 2))
            s3 = f((axis[i] + skreflengd * s2 / 2))
            s4 = f((axis[i] + skreflengd * s3))

            axis[i + 1] = axis[i] + (s1 + s2 * 2 + s3 * 2 + s4) / 6 * skreflengd
        return axis


class Pendulum:
    def __init__(self, L_1=2, m_1=1, L_2=2, m_2=1, L_3=2, m_3=1):
        self.L_1 = L_1
        self.m_1 = m_1
        self.L_2 = L_2
        self.m_2 = m_2
        self.L_3 = L_3
        self.m_3 = m_3

        self.g = g

    def pendulum(self, theta):
        fasti = -1 * self.g / (self.L_1)
        return np.sin(theta) * fasti

    def double_pendulum1(self, theta1, theta2, omega1, omega2):
        l1 = self.L_1
        l2 = self.L_2
        m1 = self.m_1
        m2 = self.m_2
        g = self.g
        d = theta2 - theta1

        if omega1 > 2e+50:
            omega1 = 2e+50
        if omega2 > 2e+50:
            omega2 = 2e+50

        if omega1 < -2e+50:
            omega1 = -2e+50
        if omega2 < -2e+50:
            omega2 = -2e+50

        cosd = math.cos(d)
        sind = math.sin(d)
        k1 = m2 * l1 * math.pow(omega1, 2) * sind * cosd
        k2 = m2 * g * math.sin(theta2) * cosd
        k3 = m2 * l2 * math.pow(omega2, 2) * sind
        k4 = -((m1 + m2) * g * math.sin(theta1))
        k5 = ((m1 + m2) * l1 - m2 * l1 * cosd * cosd)

        if l1 == 0 or k5 == 0 or (k1 + k2 + k3 + k4) == 0:
            return 0

        theta1_2prime = (k1 + k2 + k3 + k4) / k5
        return theta1_2prime

    def double_pendulum2(self, theta1, theta2, omega1, omega2):
        l1 = self.L_1
        l2 = self.L_2
        m1 = self.m_1
        m2 = self.m_2
        g = self.g
        d = theta2 - theta1

        if omega1 > 2e+50:
            omega1 = 2e+50
        if omega2 > 2e+50:
            omega2 = 2e+50

        if omega1 < -2e+50:
            omega1 = -2e+50
        if omega2 < -2e+50:
            omega2 = -2e+50

        cosd = math.cos(d)
        sind = math.sin(d)

        k1 = -m2 * l2 * math.pow(omega2, 2) * sind * cosd
        k2 = ((m1 + m2) * g * math.sin(theta1)) * cosd
        k3 = -(m1 + m2) * l1 * math.pow(omega1, 2) * sind
        k4 = -(m1 + m2) * g * math.sin(theta2)
        k5 = (m1 + m2) * l2 - m2 * l2 * cosd

        if l2 == 0 or k5 == 0 or (k1 + k2 + k3 + k4) == 0:
            return 0

        theta2_2prime = (k1 + k2 + k3 + k4) / k5
        return theta2_2prime

    def triple(self, theta1, theta2, theta3, omega1, omega2, omega3):

        # við gerðum alla handavinnuna, en fengum svo vin okkar chatgpt til þess að optimiza kóðann okkar
        # credit due fyrir openai fyrir það!

        theta1 %= np.pi * 2
        theta2 %= np.pi * 2
        theta3 %= np.pi * 2

        m1 = self.m_1
        m2 = self.m_2
        m3 = self.m_3
        L_1 = self.L_1
        L_2 = self.L_2
        L_3 = self.L_3

        theta1_theta2 = theta1 - theta2
        theta1_theta3 = theta1 - theta3
        theta2_theta3 = theta2 - theta3
        theta2_theta1 = theta2 - theta1
        theta3_theta1 = theta3 - theta1
        theta3_theta2 = theta3 - theta2

        # Pre-calculate the cosines of the differences between the angles of each pair of pendulums
        cos_theta1_theta2 = math.cos(theta1_theta2)
        cos_theta1_theta3 = math.cos(theta1_theta3)
        cos_theta2_theta3 = math.cos(theta2_theta3)
        cos_theta2_theta1 = math.cos(theta2_theta1)

        # Pre-calculate the sines of the angles of each pendulum
        sin_theta1 = math.sin(theta1)
        sin_theta2 = math.sin(theta2)
        sin_theta3 = math.sin(theta3)

        # Pre-calculate the cosines of the differences between the angles of each pair of pendulums
        sin_theta1_theta2 = math.sin(theta1_theta2)
        sin_theta1_theta3 = math.sin(theta1_theta3)
        sin_theta2_theta3 = math.sin(theta2_theta3)
        sin_theta2_theta1 = math.sin(theta2_theta1)
        sin_theta3_theta1 = math.sin(theta3_theta1)
        sin_theta3_theta2 = math.sin(theta3_theta2)

        # Pre-calculate the terms that are used multiple times in Afylki and bfylki
        m1_plus_m2_plus_m3 = m1 + m2 + m3
        m2_plus_m3 = m2 + m3
        m2_plus_m3_times_L1_times_L2 = m2_plus_m3 * L_1 * L_2
        m3_times_L1_times_L2 = m3 * L_1 * L_2
        m3_times_L1_times_L3 = m3 * L_1 * L_3
        m3_times_L2_times_L3 = m3 * L_2 * L_3
        m1_sin_theta1 = m1 * sin_theta1
        m2_sin_theta1 = m2 * sin_theta1
        m3_sin_theta1 = m3 * sin_theta1
        m3_times_omega1_times_omega2_times_sin_theta1_theta2 = m3_times_L1_times_L2 * omega1 * omega2 * sin_theta1_theta2
        m3_times_omega1_times_omega3_times_sin_theta1_theta3 = m3_times_L1_times_L3 * omega1 * omega3 * sin_theta1_theta3
        m3_times_omega2_times_omega3_times_sin_theta2_theta3 = m3_times_L2_times_L3 * omega2 * omega3 * sin_theta2_theta3

        Afylki = np.array([[L_1 * L_1 * m1_plus_m2_plus_m3, m2_plus_m3_times_L1_times_L2 * cos_theta1_theta2,
                            m3_times_L1_times_L3 * cos_theta1_theta3],
                           [m2_plus_m3_times_L1_times_L2 * cos_theta2_theta1, m2_plus_m3 * L_2 * L_2,
                            m3_times_L2_times_L3 * cos_theta2_theta3],
                           [m3_times_L1_times_L3 * cos_theta1_theta3, m3_times_L2_times_L3 * cos_theta2_theta3,
                            m3 * L_3 * L_3]])

        bfylki = np.array([[g * L_1 * (m1_sin_theta1 + m2_sin_theta1 + m3_sin_theta1)
                            + m2 * L_1 * L_2 * sin_theta1_theta2 * omega1 * omega2
                            + m3_times_omega1_times_omega3_times_sin_theta1_theta3
                            + m3_times_omega1_times_omega2_times_sin_theta1_theta2
                            + m2 * L_1 * L_2 * sin_theta2_theta1 * (omega1 - omega2) * omega2
                            + m3_times_L1_times_L2 * sin_theta2_theta1 * (omega1 - omega2) * omega2
                            + m3_times_L1_times_L3 * sin_theta3_theta1 * (omega1 - omega3) * omega3],

                           [g * L_2 * (m2 * sin_theta2 + m3 * sin_theta2)
                            + omega1 * omega2 * L_1 * L_2 * sin_theta2_theta1 * (m2 + m3)
                            + m3_times_omega2_times_omega3_times_sin_theta2_theta3
                            + (m2 + m3) * L_1 * L_2 * sin_theta2_theta1 * (omega1 - omega2) * omega1
                            + m3_times_L2_times_L3 * sin_theta3_theta2 * (omega2 - omega3) * omega3],

                           [g * L_3 * m3 * sin_theta3
                            - m3_times_omega2_times_omega3_times_sin_theta2_theta3
                            - m3_times_omega1_times_omega3_times_sin_theta1_theta3
                            + m3_times_L1_times_L3 * sin_theta3_theta1 * (omega1 - omega3) * omega1
                            + m3_times_L2_times_L3 * sin_theta3_theta2 * (omega2 - omega3) * omega2]])
        bfylki *= -1
        return la.solve(Afylki, bfylki)

    def hnitforanimationusingEuler(self, fall, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20, dempunarstuðull=0):
        follin = Foll()
        y = follin.euler(f=fall, horn=horn, hornhradi=hornhradi, fjoldiskrefa=fjoldiskrefa, lengd=lengd, dempunarstuðull=dempunarstuðull)
        hnit = []
        for theta in y:
            hnit.append(self.hornTohnit(theta))
        hnit = np.array(hnit)
        return hnit, y

    def hnitforanimationusingRK(self, fall, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20, dempunarstuðull=0):
        follin = Foll()
        y = follin.RKmethod(f=fall, horn=horn, hornhradi=hornhradi, fjoldiskrefa=fjoldiskrefa, lengd=lengd, dempunarstuðull=dempunarstuðull)
        hnit = []
        for theta in y:
            hnit.append(self.hornTohnit(theta))
        hnit = np.array(hnit)
        y = np.array(y) * (180 / np.pi)
        return hnit, y

    def hnitforanimationusingRK2(self, L_1=2, m_1=1, L_2=2, m_2=1, horn1=np.pi * 3 / 4, horn2=np.pi * 6 / 4,
                                 hornhradi1=1, hornhradi2=0, fjoldiskrefa=20 * 1000, lengd=20, dempunarstuðull=0):
        follin = Foll()
        p = Pendulum(L_1=L_1, m_1=m_1, L_2=L_2, m_2=m_2)
        arr = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=horn1, horn2=horn2,
                               hornhradi1=hornhradi1, hornhradi2=hornhradi2, fjoldiskrefa=fjoldiskrefa, lengd=lengd,
                               dempunarstuðull=dempunarstuðull)

        y1 = arr[:, 0]
        y2 = arr[:, 1]

        hnitsenior = []
        hnitjunior = []

        for theta in y1:
            hnitsenior.append(p.hornTohnit(theta))
        for index, theta in enumerate(y2):
            hnitjunior.append(p.hornTohnitjunior(y1[index], theta))

        hnitsenior = np.array(hnitsenior)
        hnitjunior = np.array(hnitjunior)
        y1 = np.array(y1) * (180 / np.pi)
        y2 = np.array(y2) * (180 / np.pi)
        return hnitsenior, hnitjunior, y1, y2

    def hnitforanimationusingRK3(self, horn1=np.pi * 3 / 4, horn2=np.pi * 6 / 4, horn3=np.pi * 6 / 4,
                                 hornhradi1=1, hornhradi2=0, hornhradi3=0, fjoldiskrefa=20 * 1000, lengd=20,
                                 dempunarstuðull=0):
        follin = Foll()
        p = Pendulum(L_1=self.L_1, m_1=self.m_1, L_2=self.L_2, m_2=self.m_2, L_3=self.L_3, m_3=self.m_3)
        arr = follin.RKmethod3(triplediple=p.triple, horn1=horn1, horn2=horn2, horn3=horn3,
                               hornhradi1=hornhradi1, hornhradi2=hornhradi2, hornhradi3=hornhradi3,
                               fjoldiskrefa=fjoldiskrefa, lengd=lengd, dempunarstuðull=dempunarstuðull)

        y1 = arr[:, 0]
        y2 = arr[:, 1]
        y3 = arr[:, 2]

        hnitsenior = []
        hnitjunior = []
        hnitjuniorjunior = []

        for horn in y1:
            hnitsenior.append(p.hornTohnit(th=horn))
        for index, horn in enumerate(y2):
            hnitjunior.append(p.hornTohnitjunior(th=y1[index], th2=horn))
        for index, horn in enumerate(y3):
            hnitjuniorjunior.append(p.hornTohnitjuniorjunior(th1=y1[index], th2=y2[index], th3=horn))

        hnitsenior = np.array(hnitsenior)
        hnitjunior = np.array(hnitjunior)
        hnitjuniorjunior = np.array(hnitjuniorjunior)

        return hnitsenior, hnitjunior, hnitjuniorjunior, y1, y2, y3

    def hornTohnit(self, th):
        L_1 = self.L_1
        return L_1 * np.sin(th), -L_1 * np.cos(th)

    def hornTohnitjunior(self, th, th2):
        L_1 = self.L_1
        L_2 = self.L_2
        return L_1 * np.sin(th) + L_2 * np.sin(th2), -L_1 * np.cos(th) - L_2 * np.cos(th2)

    def hornTohnitjuniorjunior(self, th1, th2, th3):
        L_1 = self.L_1
        L_2 = self.L_2
        L_3 = self.L_3
        return L_1 * np.sin(th1) + L_2 * np.sin(th2) + L_3 * np.sin(th3), -(L_1 * np.cos(th1) + L_2 * np.cos(th2) + L_3 * np.cos(th3))

    def create_animation2dfyrir4(self, data1, data2=None, fjoldipendula=1, title=None):
        # initializing a figure in
        # which the graph will be plotted
        # marking the x-axis and y-axis
        staerdramma = self.L_2 + self.L_1 + 3
        plt.axis('equal')

        plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))
        if fjoldipendula == 1:
            for index in range(0, data1.shape[0], 80):
                plt.clf()
                plt.title(label=title)

                x = data1[index, 0]
                y = data1[index, 1]

                x2 = data2[index, 0]
                y2 = data2[index, 1]

                plt.xticks([])
                plt.yticks([])

                plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))

                plt.xlabel('Staðsetning á x-ás')
                plt.ylabel('Staðsetning á y-ás')

                plt.plot([-staerdramma, staerdramma], [0, 0], lw=4, c="black")

                plt.scatter(x, y, lw=20, c="orange")
                plt.plot([0, x], [0, y], lw=5, c="blue", alpha=1)
                plt.scatter(x2, y2, lw=20, c="red")
                plt.plot([0, x2], [0, y2], lw=5, c="green")
                plt.pause(0.001)
        plt.show()

    def create_animation2d(self, data1, data2=None, fjoldipendula=1, title=None, savegif=False, trace=True, nafn=""):

        # initializing a figure in
        # which the graph will be plotted
        # marking the x-axis and y-axis
        staerdramma = self.L_2 + self.L_1 + 3

        bufs = []

        if fjoldipendula == 1:
            for index in range(0,data1.shape[0], 80):
                plt.clf()
                plt.title(label=title)
                x = data1[index, 0]
                y = data1[index, 1]
                plt.xticks([])
                plt.yticks([])
                plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))
                plt.xlabel('Staðsetning á x-ás')
                plt.ylabel('Staðsetning á y-ás')

                plt.plot([-staerdramma, staerdramma], [0, 0], lw=4, c="black")

                if trace:
                    plt.scatter(x, y, lw=20, c="orange")
                    plt.plot([0, x], [0, y], lw=5, c="blue")

                plt.pause(0.001)

        elif fjoldipendula == 2:
            for index in range(0, data1.shape[0], 40):
                plt.clf()
                plt.title(label=title)

                x1 = data1[index, 0]
                y1 = data1[index, 1]

                plt.xticks([])
                plt.yticks([])

                x2 = data2[index, 0]
                y2 = data2[index, 1]

                plt.xticks([])
                plt.yticks([])
                plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))

                plt.xlabel('Staðsetning á x-ás')
                plt.ylabel('Staðsetning á y-ás')

                x1plot = data1[0:index, 0]
                y1plot = data1[0:index, 1]

                x2plot = data2[0:index, 0]
                y2plot = data2[0:index, 1]

                plt.plot([-staerdramma * 2, staerdramma * 2], [0, 0], lw=3, c="black")
                if trace:
                    plt.plot(x1plot, y1plot, c="yellow")
                    plt.plot(x2plot, y2plot, c="cyan")

                plt.plot([0, x1], [0, y1], lw=5, c="blue")
                plt.plot([x1, x2], [y1, y2], lw=5, c="green")

                plt.scatter(x1, y1, lw=self.m_1 * 5, c="orange")
                plt.scatter(x2, y2, lw=self.m_2 * 5, c="red")
                if savegif:
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    bufs.append(Image.open(buf))
                else:
                    plt.pause(0.001)

                print(f"{index / data1.shape[0] * 100:.00f}%", end="\n", flush=True)

        if savegif:
            filename = str(nafn) + str(random.randint(0, 10000)) + "_gif.gif"
            f = os.path.join(os.getcwd(), filename)
            bufs[0].save(f, save_all=True, append_images=bufs[1:], optimize=False, duration=10)
        if not savegif:
            plt.show()

    def create_animation2ex2(self, data1, data2, data3, data4, fjoldipendula=1, title=None, savegif=False, offset=5, nafn=""):

        # initializing a figure in
        # which the graph will be plotted
        # marking the x-axis and y-axis
        staerdramma = self.L_2 * 2 + self.L_1 * 2 + 3

        bufs = []
        for index in range(0, data1.shape[0], 80):
            plt.clf()
            plt.title(label=title)

            x1 = data1[index, 0] + offset
            y1 = data1[index, 1]

            x2 = data2[index, 0] + offset
            y2 = data2[index, 1]

            x3 = data3[index, 0] - offset
            y3 = data3[index, 1]

            x4 = data4[index, 0] - offset
            y4 = data4[index, 1]
            plt.xticks([])
            plt.yticks([])
            plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))

            plt.xlabel('Staðsetning á x-ás')
            plt.ylabel('Staðsetning á y-ás')

            x1plot = data1[0:index, 0]
            x1plot = x1plot + offset
            y1plot = data1[0:index, 1]

            x2plot = data2[0:index, 0]
            x2plot = x2plot + offset
            y2plot = data2[0:index, 1]

            x3plot = data3[0:index, 0]
            x3plot = x3plot - offset
            y3plot = data3[0:index, 1]

            x4plot = data4[0:index, 0]
            x4plot = x4plot - offset
            y4plot = data4[0:index, 1]

            plt.plot([-staerdramma * 2, staerdramma * 2], [0, 0], lw=3, c="black")

            plt.plot(x1plot, y1plot)
            plt.plot(x2plot, y2plot)
            plt.plot(x3plot, y3plot)
            plt.plot(x4plot, y4plot)

            plt.plot([offset, x1], [0, y1], lw=5, c="blue")
            plt.plot([x1, x2], [y1, y2], lw=5, c="blue")
            plt.plot([-offset, x3], [0, y3], lw=5, c="blue")
            plt.plot([x3, x4], [y3, y4], lw=5, c="blue")

            plt.scatter(x1, y1, lw=self.m_1 * 5 * 2, c="orange")
            plt.scatter(x2, y2, lw=self.m_2 * 5 * 2, c="orange")
            plt.scatter(x3, y3, lw=self.m_1 * 5 * 2, c="orange")
            plt.scatter(x4, y4, lw=self.m_2 * 5 * 2, c="orange")

            if savegif:
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                bufs.append(Image.open(buf))
            else:
                plt.pause(0.001)
            print(f"{index / data1.shape[0] * 100:.00f}%", end="\n", flush=True)
        if savegif:
            filename = str(nafn) + str(random.randint(0, 10000)) + "_gif.gif"
            f = os.path.join(os.getcwd(), filename)
            bufs[0].save(f, save_all=True, append_images=bufs[1:], optimize=False, duration=10)
            plt.close()
        plt.show()

    def create_animation3d(self, data1, data2, data3, title=None):
        staerdramma = self.L_2 * 2 + self.L_1 * 2 + self.L_3 + 3
        bufs = []
        for index in range(0, data1.shape[0], 2):
            plt.clf()
            plt.title(label=title)

            x1 = data1[index, 0]
            y1 = data1[index, 1]

            x2 = data2[index, 0]
            y2 = data2[index, 1]

            x3 = data3[index, 0]
            y3 = data3[index, 1]

            plt.xticks([])
            plt.yticks([])
            plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))

            plt.title(r"Þrefaldur pendúll")
            plt.xlabel('Staðsetning á x-ás')
            plt.ylabel('Staðsetning á y-ás')

            x1plot = data1[0:index, 0]
            y1plot = data1[0:index, 1]

            x2plot = data2[0:index, 0]
            y2plot = data2[0:index, 1]

            x3plot = data3[0:index, 0]
            y3plot = data3[0:index, 1]

            plt.plot([-staerdramma * 2, staerdramma * 2], [0, 0], lw=3, c="black")

            plt.plot(x1plot, y1plot, alpha=0.5)
            plt.plot(x2plot, y2plot, alpha=0.5)
            plt.plot(x3plot, y3plot, alpha=0.5)

            plt.plot([0, x1], [0, y1], lw=5, c="blue")
            plt.plot([x1, x2], [y1, y2], lw=5, c="blue")
            plt.plot([x2, x3], [y2, y3], lw=5, c="blue")

            plt.scatter(x1, y1, lw=self.m_1 * 5 * 2, c="orange")
            plt.scatter(x2, y2, lw=self.m_2 * 5 * 2, c="orange")
            plt.scatter(x3, y3, lw=self.m_3 * 5 * 2, c="orange")

            plt.pause(0.001)

            print(f"{index / data1.shape[0] * 100:.00f}%", end="\n", flush=True)
            if True:
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                bufs.append(Image.open(buf))
            else:
                plt.pause(0.001)
            print(f"{index / data1.shape[0] * 100:.00f}%", end="\n", flush=True)
        if True:
            filename = "pendull" + str(random.randint(0, 10000)) + "_gif.gif"
            f = os.path.join(os.getcwd(), filename)
            bufs[0].save(f, save_all=True, append_images=bufs[1:], optimize=False, duration=10)
            plt.close()
        plt.show()
