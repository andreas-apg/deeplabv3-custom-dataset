import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import datetime
import pytz

def get_mail_server():
    mail = np.loadtxt("../utils/mail.csv", delimiter=",", dtype=str)
    mail_server = {
                'server_address':'smtp.gmail.com',
                'server_port':465,
                'user':mail[0],
                'password':mail[1]
                }
    return mail_server

def get_mail_addresses():
    mail_addresses = []
    with open('../utils/receiver.txt','r') as mails_file:
        for mail in mails_file:
            if not mail == '' and not mail == '\n':
                mail_addresses.append(mail.split('\n')[0])

    return mail_addresses

def send_mail(server, port, user, password, to, subject, body):
    msg = MIMEMultipart()
    msg['From'] = user
    msg['Subject'] = subject
    msg.attach(MIMEText(body, _subtype='plain', _charset='UTF-8'))
    smtp = smtplib.SMTP_SSL(server,port)
    smtp.login(user,password)
    smtp.sendmail(user,to,msg.as_string())

    smtp.quit()

def get_time():
    utc_dt = datetime.datetime.now(datetime.timezone.utc) # UTC time
    local_time = utc_dt.astimezone(pytz.timezone('Brazil/East')).strftime("%Y/%m/%d, %X") # string in UTC-3
    return local_time
