from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mysql.connector
from sqlalchemy import create_engine

dbku = mysql.connector.connect(
    host = '127.0.0.1',
    port = 3306,
    user = 'root',
    passwd = 'abcdefgh',
    database = 'mushroom'
)

kursor = dbku.cursor()

engine = create_engine('mysql+mysqlconnector://root:abcdefgh@127.0.0.1/mushroom', echo=False)
dbc = engine.connect()

dictcapshape = {0:'bell',1:'conical',2:'convex',3:'flat',4:'knobbed',5:'sunken'}
dictcapsurface = {0:'fibrous',1:'grooves',2:'scaly',3:'smooth'}
dictcapcolor = {0:'brown',1:'buff',2:'cinnamon',3:'gray',4:'green',5:'pink',6:'purple',7:'red',8:'white',9:'yellow'}
dictbruises = {0:'bruises',1:'no bruises'}
dictodor = {0:'almond',1:'anise',2:'creosote',3:'fishy',4:'foul',5:'musty',6:'none',7:'pungent',8:'spicy'}
dictgillattachment = {0:'attached',1:'descending',2:'free',3:'notched'}
dictgillspacing = {0:'close',1:'crowded',2:'distant'}
dictgillsize = {0:'broad',1:'narrow'}
dictgillcolor = {0:'black',1:'brown',2:'buff',3:'chocolate',4:'gray',5:'green',6:'orange',7:'pink',8:'purple',9:'red',10:'white',11:'yellow'}
dictstalkshape = {0:'enlarging',1:'tapering'}
dictstalkroot = {0:'bulbous',1:'club',2:'cup',3:'equal',4:'rhizomorphs',5:'rooted',6:'missing'}
dictstalksurfaceabovering = {0:'fibrous',1:'scaly',2:'silky',3:'smooth'}
dictstalksurfacebelowring = {0:'fibrous',1:'scaly',2:'silky',3:'smooth'}
dictstalkcolorabovering = {0:'brown',1:'buff',2:'cinnamon',3:'gray',4:'orange',5:'pink',6:'red',7:'white',8:'yellow'}
dictstalkcolorbelowring = {0:'brown',1:'buff',2:'cinnamon',3:'gray',4:'orange',5:'pink',6:'red',7:'white',8:'yellow'}
dictveilcolor = {0:'brown',1:'orange',2:'white',3:'yellow'}
dictringnumber = {0:'none',1:'one',2:'two'}
dictringtype = {0:'cobwebby',1:'evanescent',2:'flaring',3:'large',4:'none',5:'pendant',6:'sheathing',7:'zone'}
dictsporeprintcolor = {0:'black',1:'brown',2:'buff',3:'chocolate',4:'green',5:'orange',6:'purple',7:'white',8:'yellow'}
dictpopulation = {0:'abundant',1:'clustered',2:'numerous',3:'scattered',4:'several',5:'solitary'}
dicthabitat = {0:'grasses',1:'leaves',2:'meadows',3:'paths',4:'urban',5:'waste',6:'woods'}

app = Flask(__name__)

with open('database.json') as dataku:
    data = json.load(dataku)

user = []

@app.route('/', methods = ['GET', 'POST'])
def welcome():
    if str(user) != '[]':
        return redirect('/main')
    else:
        return redirect('login')

@app.route('/login', methods = ['GET', 'POST'])
def login():
    if request.method == 'POST':
        nam_l = request.form['nama_login']
        pwd_l = request.form['pass_login']
        if str(data) == '[]':
            return render_template('loginerror.html', nama = nam_l)
        else:
            for a in range(len(data)):
                if nam_l == data[a]['nama'] and pwd_l == data[a]['pass']:
                    user.append(str(nam_l))
                    return redirect('/main')
                elif a == len(data) - 1:
                    return render_template('loginerror.html')
                else:
                    continue
    else:
        if str(user) != '[]':
            return redirect('/main')
        else:
            return render_template('login.html')

@app.route('/signup', methods = ['GET', 'POST'])
def signup():
    if request.method == 'POST':
        nam_s = request.form['nama_signup']
        pwd_s = request.form['pass_signup']
        lis = []
        for i in range(len(data)):
            lis.append(data[i]['nama'])
        if str(nam_s) not in lis:
            data.append({'nama': nam_s, 'pass': pwd_s})
            y = json.dumps(data)

            json_data = open('database.json', 'w')
            json_data.write(y)
            querydb = f"""create table {str(nam_s)} (
                no int not null auto_increment,
                results varchar(50) not null,
                tested_at timestamp default current_timestamp,
                cap_shape varchar(50) not null,
                cap_surface varchar(50) not null,
                cap_color varchar(50) not null,
                bruises varchar(50) not null,
                odor varchar(50) not null,
                gill_attachment varchar(50) not null,
                gill_spacing varchar(50) not null,
                gill_size varchar(50) not null,
                gill_color varchar(50) not null,
                stalk_shape varchar(50) not null,
                stalk_root varchar(50) not null,
                stalk_surface_above_ring varchar(50) not null,
                stalk_surface_below_ring varchar(50) not null,
                stalk_color_above_ring varchar(50) not null,
                stalk_color_below_ring varchar(50) not null,
                veil_color varchar(50) not null,
                ring_number varchar(50) not null,
                ring_type varchar(50) not null,
                spore_print_color varchar(50) not null,
                population varchar(50) not null,
                habitat varchar(50) not null,
                primary key(no))"""
            kursor.execute(querydb)
            dbku.commit()
            return redirect('/')
        else:
            return render_template('signupexists.html')
    else:
        if str(user) != '[]':
            return redirect('/main')
        else:
            return render_template('signup.html')

@app.route('/main', methods = ['GET','POST'])
def home():
    if request.method == 'GET':
        if str(user) != '[]':
            return render_template('home.html', user=user[0])
        else:
            return redirect('login')

@app.route('/res', methods = ['GET','POST'])
def res():
    if request.method == 'POST':
        cap_shape = int(request.form['cap-shape'])
        cap_surface = int(request.form['cap-surface'])
        cap_color = int(request.form['cap-color'])
        bruises = int(request.form['bruises'])
        odor = int(request.form['odor'])
        gill_attachment = int(request.form['gill-attachment'])
        gill_spacing = int(request.form['gill-spacing'])
        gill_size = int(request.form['gill-size'])
        gill_color = int(request.form['gill-color'])
        stalk_shape = int(request.form['stalk-shape'])
        stalk_root = int(request.form['stalk-root'])
        stalk_surface_above_ring = int(request.form['stalk-surface-above-ring'])
        stalk_surface_below_ring = int(request.form['stalk-surface-below-ring'])
        stalk_color_above_ring = int(request.form['stalk-color-above-ring'])
        stalk_color_below_ring = int(request.form['stalk-color-below-ring'])
        veil_color = int(request.form['veil-color'])
        ring_number = int(request.form['ring-number'])
        ring_type = int(request.form['ring-type'])
        spore_print_color = int(request.form['spore-print-color'])
        population = int(request.form['population'])
        habitat = int(request.form['habitat'])

        mushrooms = [
            cap_shape,
            cap_surface,
            cap_color,
            bruises,
            odor,
            gill_attachment,
            gill_spacing,
            gill_size,
            gill_color,
            stalk_shape,
            stalk_root,
            stalk_surface_above_ring,
            stalk_surface_below_ring,
            stalk_color_above_ring,
            stalk_color_below_ring,
            veil_color,
            ring_number,
            ring_type,
            spore_print_color,
            population,
            habitat
        ]
        classpredict = model.predict([mushrooms])[0]
        results = ''
        if str(classpredict) == '0':
            results = 'Non-Poisonous'
        else:
            results = 'Poisonous'

        querydb = f"""insert into {str(user[0])}(
                results,
                cap_shape,
                cap_surface,
                cap_color,
                bruises,
                odor,
                gill_attachment,
                gill_spacing,
                gill_size,
                gill_color,
                stalk_shape,
                stalk_root,
                stalk_surface_above_ring,
                stalk_surface_below_ring,
                stalk_color_above_ring,
                stalk_color_below_ring,
                veil_color,
                ring_number,
                ring_type,
                spore_print_color,
                population,
                habitat
            ) values (
                '{str(results)}',
                '{str(dictcapshape[cap_shape])}',
                '{str(dictcapsurface[cap_surface])}',
                '{str(dictcapcolor[cap_color])}',
                '{str(dictbruises[bruises])}',
                '{str(dictodor[odor])}',
                '{str(dictgillattachment[gill_attachment])}',
                '{str(dictgillspacing[gill_spacing])}',
                '{str(dictgillsize[gill_size])}',
                '{str(dictgillcolor[gill_color])}',
                '{str(dictstalkshape[stalk_shape])}',
                '{str(dictstalkroot[stalk_root])}',
                '{str(dictstalksurfaceabovering[stalk_surface_above_ring])}',
                '{str(dictstalksurfacebelowring[stalk_surface_below_ring])}',
                '{str(dictstalkcolorabovering[stalk_color_above_ring])}',
                '{str(dictstalkcolorbelowring[stalk_color_below_ring])}',
                '{str(dictveilcolor[veil_color])}',
                '{str(dictringnumber[ring_number])}',
                '{str(dictringtype[ring_type])}',
                '{str(dictsporeprintcolor[spore_print_color])}',
                '{str(dictpopulation[population])}',
                '{str(dicthabitat[habitat])}'
            )"""
        kursor.execute(querydb)
        dbku.commit()
    
        if results == 'Non-Poisonous':
            return render_template('resultssafe.html', data=results, user=user[0])
        else:
            return render_template('resultsunsafe.html', data=results, user=user[0])
    elif request.method == 'GET':
        if str(user) == '[]':
            return redirect('login')
        else:
            return render_template('gotomainpage.html',user=user[0])

@app.route('/aboutmushrooms', methods=['GET','POST'])
def about():
    if str(user) == '[]':
        return redirect('login')
    else:
        return redirect('/aboutmushrooms/etymology')

@app.route('/aboutmushrooms/', methods=['GET','POST'])
def about2():
    if str(user) == '[]':
        return redirect('login')
    else:
        return redirect('/aboutmushrooms/etymology')

@app.route('/aboutmushrooms/etymology', methods=['GET','POST'])
def aboutetymology():
    if str(user) == '[]':
        return redirect('login')
    else:
        return render_template('aboutetymology.html',user=user[0])

@app.route('/aboutmushrooms/biology', methods=['GET','POST'])
def aboutbiology():
    if str(user) == '[]':
        return redirect('login')
    else:
        return render_template('aboutbiology.html',user=user[0])

@app.route('/aboutmushrooms/usage', methods=['GET','POST'])
def aboutusage():
    if str(user) == '[]':
        return redirect('login')
    else:
        return render_template('aboutusage.html',user=user[0])

@app.route('/logout', methods = ['GET', 'POST'])
def logout():
    user.clear()
    return redirect('/')

@app.route('/history', methods = ['GET', 'POST'])
def test():
    if str(user) == '[]':
        return redirect('login')
    else:
        df = pd.read_sql(f"select * from {str(user[0])}", dbc)
        return render_template('history.html', user=user[0], data1=df.columns.tolist(), data2=df.values.tolist())

@app.errorhandler(404)
def notfound(error):
    if str(user) != '[]':
        return render_template('errorhandler1.html', user=user[0])
    else:
        return render_template('errorhandler2.html')

if __name__ == '__main__':
    df = pd.read_csv('mushrooms.csv')
    model = joblib.load('modelmushroom')
    app.run(debug = True)