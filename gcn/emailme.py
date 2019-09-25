# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:44:51 2018

@author: admin
"""
import os
path=os.path.abspath(os.path.dirname(__file__))
os.chdir(path)
import time

import smtplib    
from email.mime.multipart import MIMEMultipart    
from email.mime.text import MIMEText    
from email.mime.image import MIMEImage 
from email.header import Header 
import  socket
import getpass

#send an email when exps finished
def email(receiver='1016064797@qq.com',smtpserver = 'smtp.qq.com',
          username = '1291996074@qq.com',password='heoayjfggwmehhie',sender='1291996074@qq.com',
          sender_name='Ziang',subject = 'Python email test',text="Hi!\n I am little server",
          images=None, files=None):
    
    
    msg = MIMEMultipart('mixed') 
    msg['Subject'] = subject
    msg['From'] = '{} <{}>'.format(sender,sender)
    msg['To'] = ";".join(receiver) 
    #msg['Date']='2012-3-16'
    
    text_plain = MIMEText(text,'plain', 'utf-8')    
    msg.attach(text_plain)    
    
    if images:
        for image in images:
            if image:
                sendimagefile=open(image,'rb').read()
                image = MIMEImage(sendimagefile)
                image.add_header('Content-ID','<image1>')
                image["Content-Disposition"] = 'attachment; filename="{}"'.format(image)
                msg.attach(image)
    
    
    if files:
        for file in files:
            if file:
                sendfile=open(file,'rb').read()
                text_att = MIMEText(sendfile, 'base64', 'utf-8') 
                text_att["Content-Type"] = 'application/octet-stream'  
               
                #text_att["Content-Disposition"] = 'attachment; filename="aaa.txt"'
                
                text_att.add_header('Content-Disposition', 'attachment', filename=file)
                
                #text_att["Content-Disposition"] = u'attachment; filename="中文附件.txt"'.decode('utf-8')
                msg.attach(text_att)    
           
    smtp = smtplib.SMTP()    
    smtp.connect(smtpserver)
    smtp.login(username, password)    
    smtp.sendmail(sender, receiver, msg.as_string())    
    smtp.quit()
    
    


def lab_email(parameters='',files=[],ExtraInfo=None):
    """
    ----------------------------------------------------------------------
    
    After the test, the results of the experiment will be delivered to your
     work mailbox by email, and the server needs to access the Internet.
    ----------------------------------------------------------------------
    """
    sender_user_name = getpass.getuser() # Get the current username
    sender_hostname = socket.gethostname() # Get the current host name
    #receiver='XXX@126.com'
    now=time.asctime()
    
    text="Test on {}@{} has successfully finished!\nTime is {}\nparameters is {}".format( sender_user_name,sender_hostname,now,parameters)
    try:
        if ExtraInfo:
            text+=str(ExtraInfo)
    except:
        pass
    email(receiver=['1016064797@qq.com'],smtpserver = 'smtp.qq.com',
          username = '1291996074@qq.com',password='heoayjfggwmehhie',sender='1291996074@qq.com',
          sender_name='Ziang',
          subject = 'Test on {}@{} has successfully finished!'.format(sender_user_name,sender_hostname),
          text=text,images=None, files=files)
    
    
    
    
    








