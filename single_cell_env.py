"""
single_cell_env.py
"""

import cv2
import numpy as np
import scipy.constants
import scipy.integrate
import time
import os

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
img = np.zeros((512,512,3), np.uint8)
laser_var=10
laser_mag=10000


def obscurity(x,y):
    """
    random obscurity of x an y
    """
    x2=1 / (1 + np.exp(-x/50))*512
    y2=1.2*y+np.exp(0.01*y)
    x2=x
    y2=y
    return x2, y2

def normal_pdf(x, mean, var):
    return np.exp(-(x - mean)**2 / (2*var))


# mouse callback function
def draw_circle(event,x,y,flags,param):
    global drawing, mode, ix, iy

    if event == cv2.EVENT_RBUTTONUP:
        print("reset")
        mode=True
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix=x
        iy=y
    elif event == cv2.EVENT_MOUSEMOVE:
        ix=x
        iy=y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing=False




def F_scatter(r, n0=1.335, a=30e-6, n1=1.37):
    """
    compute the scatering Force on a particle
    Inputs:
        n_0 (float): The refractive index of the surrounding media
        n_1 (float): the refractive index of the particle
        a (float): the size of the particle (m)
    """
    m=n1/n0
    c=scipy.constants.c
    k=scipy.constants.k
    return 8*np.pi*n0*k**4*a**6/(3*c)*((m**2-1)/(m**2+2))**2*I(r)

def I(r, mu=0,sigma=laser_var,magnitude=laser_mag):
    """
    Compute the beam intensity. Assume it is gaussian
    """
    return magnitude*np.exp(-(r-mu)**2/(2*sigma**2))


class point():
    def __init__(self, m=0.5, b=1, k=0, x=0, y=0, dt=1):
        # print(x)
        self.x=x
        # print(self.x)
        self.y=y
        self.dx=1
        self.dy=1
        self.ddx=0
        self.ddy=0
        self.dt=dt
        self.m=m #2*10^-12
        self.b=b #0.8882/100/10
        self.k=k
        self.time=0
        self.history=[]

    def update(self, dt, Fx, Fy):
        self.dt=dt
        # print(Fx,Fy, self.x,self.y)
        def dynamic_x(X, t, m, b, k, Fx):
            x, dx=X
            dXdt=[dx,-b/m*dx-k/m*x+Fx]
            return dXdt # velocity,acceleration

        def dynamic_y(Y, t, m, b, k, Fy):
            y, dy=Y
            dYdt=[dy, -b/m*dy-k/m*y+Fy]
            return dYdt
        a_t=np.arange(0,self.dt, self.dt/5)

        sol_x=scipy.integrate.odeint(dynamic_x, [self.x, self.dx], a_t, args=(self.m, self.b, self.k, Fx))
        sol_y=scipy.integrate.odeint(dynamic_y, [self.y, self.dy], a_t, args=(self.m, self.b, self.k, Fy))

        # print(Fx,self.dx,Fy,self.dy)

        # print(Fx,Fy)
        self.dx=sol_x[-1,1]
        self.x=sol_x[-1,0]
        self.dy=sol_y[-1,1]
        self.y=sol_y[-1,0]

    def plot_point(self, img):
        # print(self.x,self.y)
        img=cv2.circle(img,(int(self.x),int(self.y)),10,(255,0,255),-1)
        self.history.append((int(self.x),int(self.y)))
        if len(self.history)>1500:
            self.history=self.history[-1500:]
        for pt in self.history:
            img=cv2.circle(img,pt,1,(255,0,255),-1)
        # print(self.x)
        return img

class particle():
    def __init__(self, a=30e-6, x=0, y=0):
        self.a=a
        self.x=x
        self.y=y
        self.b=0.8882/100/10 #Pa*s
        self.m=2*10^-12
        self.vx=0
        self.vy=0
        self.ax=0
        self.ay=0
        self.history=[]
    def update(self, dt,Fx=0,Fy=0):
        #balance for x
        self.ax=-self.b/self.m*self.vx+Fx/self.m
        self.vx+=self.ax*(dt)
        self.x+=self.vx*(dt)

        # balance for y
        self.ay=-self.b/self.m*self.vy+Fy/self.m
        self.vy+=self.ay*(dt)
        self.y+=self.vy*(dt)



        # if self.vx>0:
        #     print(self.vx,self.vy)
        # if F>0:
        #     print(F/self.m*np.cos(theta))

    def plot_point(self, img):
        img=cv2.circle(img,(int(self.x*10e6),int(self.y*10e6)),10,(255,0,255),-1)
        self.history.append((int(self.x*10e6),int(self.y*10e6)))
        for pt in self.history:
            img=cv2.circle(img,pt,1,(255,0,255),-1)
        return img


def main(record=True):
    global img,ix, iy, drawing, mode,laser_mag, laser_var

    # Create a black image, a window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    # p=particle(now,x=10e-6, y=10e-6)
    if record:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out=None
        fname=os.path.join(os.getcwd(),'output.avi')
        print(fname)
    while(1):
        if mode:
            p=point(x=np.random.randint(img.shape[0]), y=np.random.randint(img.shape[1]))
            mode=False
            time_old=time.time()
            penalty=0
        if drawing==True:
            x_min=0
            y_min=0
            x_max,y_max=img.shape[:2]
            xx=np.linspace(x_min,x_max, x_max)
            yy=np.linspace(y_min,y_max, y_max)
            ix2, iy2=obscurity(ix,iy)
            gauss_x_high = normal_pdf(xx, ix2, laser_var)
            gauss_y_high = normal_pdf(yy, iy2, laser_var)
            weights=np.array(np.meshgrid(gauss_x_high, gauss_y_high)).prod(0)
            # print(np.amax(weights))
            weights=(weights*255).astype(np.uint8)
            # weights=((weights-np.amin(weights))/(np.amax(weights/np.amin(weights)))).astype(np.uint8)
            img=cv2.applyColorMap(weights, cv2.COLORMAP_OCEAN) # try spring, autumn, bone, summer, jet, rainbow
            r=np.sqrt((p.x-ix2)**2+(p.y-iy2)**2) # radial distance
            F_applied=F_scatter(r)
            F_applied=I(r)
            theta=np.arctan((p.y-iy2)/(p.x-ix2))
            dt=(time.time()-time_old)
            penalty+=(dt)*r
            time_old=time.time()
            # print(penalty)

            if (p.x<ix2):
                theta=theta+np.pi



        else:
            img=img*0
            F_applied=0
            theta=0
            dt=(time.time()-time_old)
            time_old=time.time()
        p.update(dt, Fx=-F_applied*np.cos(theta)*0.01, Fy=-F_applied*np.sin(theta)*0.01)
        img=p.plot_point(img)
        if record:
            if out==None:
                size=img.shape[:2]
                out = cv2.VideoWriter(fname,fourcc, 60.0, (size[1], size[0]))
            out.write(img)
        cv2.imshow('image',img)

        if cv2.waitKey(10) & 0xFF == 27:
            break
    if record:
        out.release()
        print('released')
    cv2.destroyAllWindows()


class opticalTweezers():
    def __init__(self, consecutive_frames=1):
        self.drawing = False # true if mouse is pressed
        self.mode = True # if True, draw rectangle. Press 'm' to toggle to curve
        # self.ix,self.iy = -1,-1
        self.img = np.zeros((512,512,3), np.uint8)
        self.laser_var=10
        self.laser_mag=10000
        # self.consecutive_frames=consecutive_frames
        # self.history=[None for i in range(self.consecutive_frames)]
        self.reset()
        self.actions_space=[0, 0]

        self.action_log=[]

        self.user_x=0
        self.user_y=0
        cv2.namedWindow('rendered')
        cv2.setMouseCallback('rendered',self.update_user)
        self.episode=0
        self.out=None

        self.action_space=[-1, -1, -1,-1]
        self.observation_space=[self.p.x, self.p.y, -1,-1,-1,-1]

    def update_user(self, event, x,y,flags, params):
        """
        for internal updates
        """
        if event ==cv2.EVENT_MOUSEMOVE:
            self.user_x=x
            self.user_y=y
    def set_user(self,user_x=0, user_y=0):
        """
        for external updates
        """
        self.user_x=user_x
        self.user_y=user_y


    def reset(self, user_x=None, user_y=None):
        self.img = np.zeros((512,512,3), np.uint8)
        self.p=point(x=np.random.randint(img.shape[0]), y=np.random.randint(img.shape[1]))
        if (user_x==None)|(user_y==None):
            user_x=np.random.randint(img.shape[0])
            user_y=np.random.randint(img.shape[1])
        self.user_x=user_x
        self.user_y=user_y
        self.ix=self.p.x+np.random.normal(0, 10,1)
        self.iy=self.p.y+np.random.normal(0, 10,1)
        self.ix=np.random.randint(img.shape[0])
        self.iy=np.random.randint(img.shape[1])
        self.dix=0
        self.diy=0
        # self.history=[[self.p.x, self.p.y, user_x, user_y] for item in self.history]
        self.time_old=0
        self.reward=0
        return np.asarray([self.p.x, self.p.y, user_x, user_y, self.ix, self.iy])

    def render(self, pt=[], text="", episode=None):
        ix, iy=obscurity(self.ix,self.iy) #

        x_min=0
        y_min=0
        x_max,y_max=self.img.shape[:2]
        xx=np.linspace(x_min,x_max, x_max)
        yy=np.linspace(y_min,y_max, y_max)
        gauss_x_high = normal_pdf(xx, ix, laser_var)
        gauss_y_high = normal_pdf(yy, iy, laser_var)
        weights=np.array(np.meshgrid(gauss_x_high, gauss_y_high)).prod(0)
        # print(np.amax(weights))
        weights=(weights*255).astype(np.uint8)
        # weights=((weights-np.amin(weights))/(np.amax(weights/np.amin(weights)))).astype(np.uint8)
        self.img=cv2.applyColorMap(weights, cv2.COLORMAP_OCEAN) # try spring, autumn, bone, summer, jet, rainbow
        if len(text)>0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img=cv2.putText(self.img,text,(10,480), font, 1,(255,255,255),1,cv2.LINE_AA)
        self.img=self.p.plot_point(self.img)
        self.img=cv2.circle(self.img,(int(self.user_x),int(self.user_y)),10,(0,0,255),1)
        if len(pt)>0:
            for p in pt:
                x,y=p
                self.img=cv2.circle(self.img,(int(x),int(y)),10,(0,255,0),1)
        cv2.imshow('rendered', self.img)
        cv2.waitKey(2)

        try:
            int(episode)
        except:
            return
        self.save_frame(text, episode)


    def save_frame(self, text, episode):
        if episode!=self.episode:
            #new video object
            if self.out==None:
                pass
            else:
                self.out.release()
                self.out=None
            self.episode=episode

        if self.out==None:
            size=self.img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fname='{}_episode-{}.avi'.format(text.split(" ")[0], str(episode).zfill(6))
            self.out = cv2.VideoWriter(fname,fourcc, 60.0, (size[1], size[0]))
        self.out.write(self.img)


    def _reward(self, action):
        dix,diy=action
        reward=0
        #add reward for moving towards agent
        reward+=0.001*(self.p.x-self.ix)*dix+0.001*(self.p.y-self.iy)*diy
        #add rewardfor moving p towards target
        r=np.sqrt((self.p.x-self.ix)**2+(self.p.y-self.iy)**2)
        reward+=normal_pdf(r, 0, 10)*((self.user_x-self.ix)*dix+(self.user_y-self.iy)*diy)
        return reward

    def step(self, action, drawing=True, dt=0.1):
        if len(action)==2:
            dix, diy=action
        elif len(action)==4:
            pos_x, neg_x, pos_y, neg_y=action
            dix=pos_x-neg_x
            diy=pos_y-neg_y
        elif len(action)==6:
            pos_x, neg_x, pos_y, neg_y, mag_x, mag_y=action
            dix=(pos_x-neg_x)*mag_x
            diy=(pos_y-neg_y)*mag_y
            # dix,diy,yes_x, yes_y=action
            # dix=-1 if dix<0.5 else 1
            # diy=-1 if diy<0.5 else 1
            # dix=0 if yes_x<0.5 else dix
            # diy=0 if yes_y<0.5 else diy





        if drawing:
            reward_baseline=dt*normal_pdf(np.sqrt((self.p.x-self.user_x)**2+(self.p.y-self.user_y)**2), 0, 10)

            #update the controller dynamics
            # def _integrate_speed(X, t, b, k, m, F):
            #     pos, vel=X
            #     dXdt = [vel, -b/m*vel-k/m*pos+F/m] #physics to move the laser 'agent'
            #     return dXdt
            #
            # t=np.linspace(0, dt, dt/5)
            # X_0=[self.ix, self.dix]
            # solx=scipy.integrate.odeint(_integrate_speed, X_0, t, args=(0.1, 0, 1, dix-0.5))
            # # print(solx)
            # self.ix=solx[-1, 0]
            # self.dix=solx[-1,1]
            #
            # Y_0=[self.iy, self.diy]
            # sol_y=scipy.integrate.odeint(self._integrate_speed, Y_0, t, args=(0.1, 0, 1, diy-0.5))
            # self.iy=sol_y[-1, 0]
            # self.diy=sol_y[-1,1]

            def dynamic_x(X, t, m, b, k, Fx):
                x, dx=X
                dXdt=[dx,-b/m*dx-k/m*x+Fx]
                return dXdt # velocity,acceleration

            def dynamic_y(Y, t, m, b, k, Fy):
                y, dy=Y
                dYdt=[dy, -b/m*dy-k/m*y+Fy]
                return dYdt
            a_t=np.arange(0,dt, dt/5)
            m=1
            b=0.5
            k=0
            Fx=(dix)#*yes_x
            Fy=(diy)#*yes_y
            sol_x=scipy.integrate.odeint(dynamic_x, [self.ix, self.dix], a_t, args=(m, b, k, Fx))
            sol_y=scipy.integrate.odeint(dynamic_y, [self.iy, self.diy], a_t, args=(m, b, k, Fy))

            # print(Fx,self.dx,Fy,self.dy)

            # print(Fx,Fy)
            self.dix=sol_x[-1,1]
            self.ix=sol_x[-1,0]
            self.diy=sol_y[-1,1]
            self.iy=sol_y[-1,0]

            # if it was only 4 values, consider discrete movement
            if len(action)==4:
                pos_x, neg_x, pos_y, neg_y=action
                self.ix=self.ix+pos_x-neg_x
                self.iy=self.iy+pos_y-neg_y

            #check to see if dynamics are violated
            # if (self.ix<0):
            #     self.dix=0
            #     self.ix=0
            # elif(self.ix>img.shape[0]):
            #     self.dix=0
            #     self.ix=img.shape[0]
            # if (self.iy<0):
            #     self.diy=0
            #     self.iy=0
            # elif(self.iy>img.shape[1]):
            #     self.diy=0
            #     self.yx=img.shape[1]


            ix, iy=obscurity(self.ix,self.iy) #

            r=np.sqrt((self.p.x-ix)**2+(self.p.y-iy)**2) # radial distance
            F_applied=F_scatter(r)
            F_applied=I(r)
            theta=np.arctan((self.p.y-iy)/(self.p.x-ix))
            self.reward=dt*normal_pdf(np.sqrt((self.p.x-self.user_x)**2+(self.p.y-self.user_y)**2), 0, 10)-reward_baseline
            if (self.p.x<ix):
                theta=theta+np.pi

        else:
            self.img=self.img*0
            F_applied=0
            theta=0
        self.p.update(dt, Fx=-F_applied*np.cos(theta)*0.01, Fy=-F_applied*np.sin(theta)*0.01)


        # #other features to return
        done=False
        # if (self.p.x>self.img.shape[0])|(self.p.x<0):
        #     done=True
        #     # self.reward-=1
        # if (self.p.y>self.img.shape[1])|(self.p.y<0):
        #     done=True
        #     # self.reward-=1
        # if (self.ix>self.img.shape[0]+10)|(self.ix<-10):
        #     done=True
            # self.reward-=1
        reward=0
        if (self.ix>self.img.shape[0]):
            self.ix=self.img.shape[0]
            self.dix=0
        if self.ix<0:
            self.ix=0
            self.dix=0
        if (self.iy>self.img.shape[1]):
            self.iy=self.img.shape[1]
            self.diy=0
        if self.iy<0:
            self.iy=0
            self.diy=0
            # self.reward-=1
        if np.sqrt((self.p.x-self.user_x)**2+(self.p.y-self.user_y)**2)<10:
            done=True
            self.reward=1
        info=None
        # self.reward=self._reward(action)
        # self.reward=float(self.reward)

        self.observation_space=np.asarray([self.p.x, self.p.y, self.user_x,self.user_y, self.ix, self.iy])
        # self.observation_space[np.where(self.observation_space<0)]=0
        # self.observation_space[np.where(self.observation_space>1)]=1
        return self.observation_space, self.reward, done, info


class opticalTweezers_modelpredict():
    def __init__(self, consecutive_frames=1):
        self.drawing = False # true if mouse is pressed
        self.mode = True # if True, draw rectangle. Press 'm' to toggle to curve
        # self.ix,self.iy = -1,-1
        self.img = np.zeros((512,512,3), np.uint8)
        self.laser_var=10
        self.laser_mag=10000
        # self.consecutive_frames=consecutive_frames
        # self.history=[None for i in range(self.consecutive_frames)]
        self.reset()
        self.actions_space=[0, 0]

        self.action_log=[]

        self.action_space=[-1, -1, -1,-1]
        self.observation_space=[self.p.x, self.p.y, -1,-1,-1,-1]


    def reset(self, user_x=None, user_y=None):
        self.img = np.zeros((512,512,3), np.uint8)
        self.p=point(x=np.random.randint(img.shape[0]), y=np.random.randint(img.shape[1]))
        if (user_x==None)|(user_y==None):
            user_x=np.random.randint(img.shape[0])
            user_y=np.random.randint(img.shape[1])
        self.user_x=user_x
        self.user_y=user_y
        self.ix=self.p.x+np.random.normal(0, 10,1)
        self.iy=self.p.y+np.random.normal(0, 10,1)
        self.dix=0
        self.diy=0
        # self.history=[[self.p.x, self.p.y, user_x, user_y] for item in self.history]
        self.time_old=0
        self.reward=0
        return np.asarray([self.p.x/512, self.p.y/512, user_x/512, user_y/512, self.ix/512, self.iy/512])

    def render(self):
        ix, iy=obscurity(self.ix,self.iy) #

        x_min=0
        y_min=0
        x_max,y_max=self.img.shape[:2]
        xx=np.linspace(x_min,x_max, x_max)
        yy=np.linspace(y_min,y_max, y_max)
        gauss_x_high = normal_pdf(xx, ix, laser_var)
        gauss_y_high = normal_pdf(yy, iy, laser_var)
        weights=np.array(np.meshgrid(gauss_x_high, gauss_y_high)).prod(0)
        # print(np.amax(weights))
        weights=(weights*255).astype(np.uint8)
        # weights=((weights-np.amin(weights))/(np.amax(weights/np.amin(weights)))).astype(np.uint8)
        self.img=cv2.applyColorMap(weights, cv2.COLORMAP_OCEAN) # try spring, autumn, bone, summer, jet, rainbow

        self.img=self.p.plot_point(self.img)
        self.img=cv2.circle(self.img,(int(self.user_x),int(self.user_y)),10,(0,0,255),1)
        cv2.imshow('rendered', self.img)
        cv2.waitKey(2)

    def _reward(self, action):
        dix,diy=action
        reward=0
        #add reward for moving towards agent
        reward+=0.001*(self.p.x-self.ix)*dix+0.001*(self.p.y-self.iy)*diy
        #add rewardfor moving p towards target
        r=np.sqrt((self.p.x-self.ix)**2+(self.p.y-self.iy)**2)
        reward+=normal_pdf(r, 0, 10)*((self.user_x-self.ix)*dix+(self.user_y-self.iy)*diy)
        return reward

    def step(self, action, drawing=True, dt=0.1):
        if len(action)==2:
            dix, diy=action
        else:
            dix,diy,yes_x, yes_y=action
            dix=-1 if dix<0.5 else 1
            diy=-1 if diy<0.5 else 1
            dix=0 if yes_x<0.5 else dix
            diy=0 if yes_y<0.5 else diy

        #m, b=model





        if drawing:
            reward_baseline=dt*normal_pdf(np.sqrt((self.p.x-self.user_x)**2+(self.p.y-self.user_y)**2), 0, 10)

            #update the controller dynamics

            def dynamic_x(X, t, m, b, k, Fx):
                x, dx=X
                dXdt=[dx,-b/m*dx-k/m*x+Fx]
                return dXdt # velocity,acceleration

            def dynamic_y(Y, t, m, b, k, Fy):
                y, dy=Y
                dYdt=[dy, -b/m*dy-k/m*y+Fy]
                return dYdt
            a_t=np.arange(0,dt, dt/5)
            m=1
            b=0.01
            k=0
            Fx=(dix)#*yes_x
            Fy=(diy)#*yes_y
            sol_x=scipy.integrate.odeint(dynamic_x, [self.ix, self.dix], a_t, args=(m, b, k, Fx))
            sol_y=scipy.integrate.odeint(dynamic_y, [self.iy, self.diy], a_t, args=(m, b, k, Fy))

            # print(Fx,self.dx,Fy,self.dy)

            # print(Fx,Fy)
            self.dix=sol_x[-1,1]
            self.ix=sol_x[-1,0]
            self.diy=sol_y[-1,1]
            self.iy=sol_y[-1,0]

            #check to see if dynamics are violated
            # if (self.ix<0):
            #     self.dix=0
            #     self.ix=0
            # elif(self.ix>img.shape[0]):
            #     self.dix=0
            #     self.ix=img.shape[0]
            # if (self.iy<0):
            #     self.diy=0
            #     self.iy=0
            # elif(self.iy>img.shape[1]):
            #     self.diy=0
            #     self.yx=img.shape[1]


            ix, iy=obscurity(self.ix,self.iy) #


            r=np.sqrt((self.p.x-ix)**2+(self.p.y-iy)**2) # radial distance
            F_applied=F_scatter(r)
            F_applied=I(r)
            theta=np.arctan((self.p.y-iy)/(self.p.x-ix))
            self.reward=dt*normal_pdf(np.sqrt((self.p.x-self.user_x)**2+(self.p.y-self.user_y)**2), 0, 10)-reward_baseline
            if (self.p.x<ix):
                theta=theta+np.pi

        else:
            self.img=self.img*0
            F_applied=0
            theta=0
        self.p.update(dt, Fx=-F_applied*np.cos(theta)*0.01, Fy=-F_applied*np.sin(theta)*0.01)
        self.reward=0

        #other features to return
        done=False
        if (self.p.x>self.img.shape[0]+10)|(self.p.x<-10):
            done=True
            # self.reward-=1
        if (self.p.y>self.img.shape[1]+10)|(self.p.y<-10):
            done=True
            # self.reward-=1
        if (ix>self.img.shape[0]+10)|(ix<-10):
            done=True
            # self.reward-=1
        if (iy>self.img.shape[1]+10)|(iy<-10):
            done=True
            # self.reward-=1
        if np.sqrt((self.p.x-self.user_x)**2+(self.p.y-self.user_y)**2)<10:
            done=True
            self.reward=1
        info=None
        # self.reward=self._reward(action)
        # self.reward=float(self.reward)

        self.observation_space=np.asarray([self.p.x/512, self.p.y/512, self.user_x/512,self.user_y/512, self.ix/512, self.iy/512])
        # self.observation_space[np.where(self.observation_space<0)]=0
        # self.observation_space[np.where(self.observation_space>1)]=1
        return self.observation_space, self.reward, done, info



if __name__=="__main__":
    main()
