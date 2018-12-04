from flask import Flask, render_template, request
import sqlite3 as sql
application = Flask(__name__)
import pandas as pd
from sklearn.cluster import KMeans
import os
import numpy as np
import numpy
from sklearn.metrics import silhouette_samples, silhouette_score

df = pd.read_csv('titanic3.csv')
df.fillna(0, inplace=True)

port = int(os.getenv('PORT', 8000))


# This function is used to load the csv file into the database
@application.route('/ctb', methods=['GET', 'POST'])
def ctb():
    if request.method == 'POST':
        f = request.files['myf']
        d = pd.read_csv(f)
        d.fillna(0, inplace=True)
        cnx = sql.connect('database.db')
        d.to_sql(name="titanic", con=cnx, if_exists="replace", index=False)
        return render_template('home.html')


@application.route('/ctb1', methods=['GET', 'POST'])
def ctb1():
    if request.method == 'POST':
        f = request.files['myfi']
        d = pd.read_csv(f)
        d.fillna(0, inplace=True)
        cnx = sql.connect('database.db')
        d.to_sql(name="earth", con=cnx, if_exists="replace", index=False)
        return render_template('home.html')

@application.route('/ctb2', methods=['GET', 'POST'])
def ctb2():
    if request.method == 'POST':
        f = request.files['myfil']
        d = pd.read_csv(f)
        d.fillna(0, inplace=True)
        cnx = sql.connect('database.db')
        d.to_sql(name="minnow", con=cnx, if_exists="replace", index=False)
        return render_template('home.html')

# ref: developers.google.com/chart/
# ref: https://pythonprogramming.net/k-means-titanic-dataset-machine-learning-tutorial/
def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df


df = handle_non_numerical_data(df)


@application.route('/')
def home():
    return render_template('home.html')


@application.route('/clusterform')
def clusterform():
    return render_template('clusterform.html')

@application.route('/qsix')
def qsix():
    return render_template('qsix.html')

@application.route('/qseven')
def qseven():
    return render_template('qseven.html')


@application.route('/clusterform1')
def clusterform1():
    return render_template('clusterform1.html')


@application.route('/images')
def images():
    return render_template('imageform.html')

@application.route('/clusterform2')
def clusterform2():
    return render_template('clusterform2.html')

# Function to display the list of all the earthquakes in the database
@application.route('/lis')
def lis():
    cnx = sql.connect("database.db")
    cnx.row_factory = sql.Row
    cr = cnx.cursor()
    cr.execute("select * from titanic")
    rows = cr.fetchall();
    return render_template("list.html", rows=rows)

@application.route('/list1')
def list1():
    cnx = sql.connect("database.db")
    cnx.row_factory = sql.Row
    cr = cnx.cursor()
    cr.execute("select * from earth")
    rows = cr.fetchall();
    return render_template("list1.html", rows=rows)

@application.route('/list2')
def list2():
    cnx = sql.connect("database.db")
    cnx.row_factory = sql.Row
    cr = cnx.cursor()
    cr.execute("select * from minnow")
    rows = cr.fetchall();
    return render_template("list2.html", rows=rows)

@application.route('/earthq',  methods=['GET','POST'])
def earthq():
    cnx = sql.connect("database.db")
    cnx.row_factory = sql.Row
    cr = cnx.cursor()
    cr.execute("select * from earth WHERE mag < 2 AND (time BETWEEN '2018-06-07T02:24:59.290Z' AND '2018-06-14T02:24:59.290Z')")
    rows = cr.fetchall();
    a = len(rows)
    cr.execute("select * from earth WHERE mag BETWEEN 2 AND 2.5 AND (time BETWEEN '2018-06-07T02:24:59.290Z' AND '2018-06-14T02:24:59.290Z') ")
    rows = cr.fetchall();
    b = len(rows)
    cr.execute("select * from earth WHERE mag BETWEEN 2.5 AND 3 AND (time BETWEEN '2018-06-07T02:24:59.290Z' AND '2018-06-14T02:24:59.290Z')")
    rows = cr.fetchall();
    c = len(rows)
    cr.execute("select * from earth WHERE mag BETWEEN 3 AND 3.5 AND (time BETWEEN '2018-06-07T02:24:59.290Z' AND '2018-06-14T02:24:59.290Z')")
    rows = cr.fetchall();
    d = len(rows)
    cr.execute("select * from earth WHERE mag BETWEEN 3.5 AND 4 AND (time BETWEEN '2018-06-07T02:24:59.290Z' AND '2018-06-14T02:24:59.290Z')")
    rows = cr.fetchall();
    e = len(rows)
    cr.execute("select * from earth WHERE mag BETWEEN 4 AND 4.5 AND (time BETWEEN '2018-06-07T02:24:59.290Z' AND '2018-06-14T02:24:59.290Z')")
    rows = cr.fetchall();
    f = len(rows)
    cr.execute("select * from earth WHERE mag BETWEEN 4.5 AND 5 AND (time BETWEEN '2018-06-07T02:24:59.290Z' AND '2018-06-14T02:24:59.290Z')")
    rows = cr.fetchall();
    g = len(rows)
    cr.execute("select * from earth WHERE mag BETWEEN 5 AND 5.5 AND (time BETWEEN '2018-06-07T02:24:59.290Z' AND '2018-06-14T02:24:59.290Z')")
    rows = cr.fetchall();
    h = len(rows)
    cr.execute("select * from earth WHERE mag BETWEEN 5.5 AND 6 AND (time BETWEEN '2018-06-07T02:24:59.290Z' AND '2018-06-14T02:24:59.290Z')")
    rows = cr.fetchall();
    i = len(rows)
    sizes = [a,b,c,d,e,f,g,h,i]
    return render_template("gapresult.html", var=sizes)


# Function to find the centers of the clusters
# REF: https://mubaris.com/2017/10/01/kmeans-clustering-in-python/
@application.route('/cluster3d', methods=['GET', 'POST'])
def cluster3d():
    data = df['cabin']
    data1 = df['age']
    data2 = df['fare']
    y = list(zip(data, data1, data2))
    lis = numpy.array(y)
    x = 5
    est = KMeans(n_clusters=x)
    est.fit(y)
    e = est.cluster_centers_
    dist = numpy.linalg.norm(e)
    # print(dist)
    labels = est.labels_
    uni, counts = numpy.unique(labels, return_counts=True)
    count = dict(zip(uni, counts))
    # print(labels)
    # print(count)
    di = []
    for i in range(0, x):
        for j in range(i + 1, x):
            d = numpy.linalg.norm(e[i] - e[j])
            di.append(d)
            print(i)
            print(j)
            print(d)
    # f = plt.figure()
    # q = mpld3.display(f)
    clusterer = KMeans(n_clusters=x, random_state=10)
    cluster_labels = clusterer.fit_predict(y)
    silhouette_avg = silhouette_score(y, cluster_labels)
    print("For n_clusters =", x, "The average silhouette_score is :", silhouette_avg)
    return render_template('cluster3d.html', rows=e, dist=dist, count=count, num=x, distance=di, sli=silhouette_avg)


# Function to find the centers of the clusters
# REF: https://mubaris.com/2017/10/01/kmeans-clustering-in-python/
@application.route('/cluster', methods=['GET', 'POST'])
def cluster():
    '''con = sql.connect("database.db")
    con.row_factory = sql.Row
    cursor = con.cursor()
    cursor.execute("select cabin, fare from titanic")
    rows = cursor.fetchall()'''
    data1 = df['age']
    data2 = df['fare']
    y = list(zip(data1, data2))
    '''d = DataFrame(rows)
    #print(d)'''
    lis = numpy.array(y)
    x = request.form['start']
    x = int(x)
    est = KMeans(n_clusters=x)
    est.fit(y)
    e = est.cluster_centers_
    dist = numpy.linalg.norm(e)
    # print(dist)
    labels = est.labels_
    uni, counts = numpy.unique(labels, return_counts=True)
    print(uni)
    print(counts)
    count = dict(zip(uni, counts))
    di = []
    for i in range(0, x):
        for j in range(i + 1, x):
            d = numpy.linalg.norm(e[i] - e[j])
            di.append(d)
            print(i)
            print(j)
            print(d)
    '''plt.scatter(lis[:,0], lis[:, 1], c = labels, cmap='rainbow')
    plt.scatter(e[:,0] ,e[:,1], color='black')
    plt.savefig('static/scatter1.png')
    plt.show()'''
    # f = plt.figure()
    # q = mpld3.display(f)
    m = max(di)
    clusterer = KMeans(n_clusters=x, random_state=10)
    cluster_labels = clusterer.fit_predict(y)
    silhouette_avg = silhouette_score(y, cluster_labels)
    return render_template('cluster.html', rows=e, dist=dist, count=count, num=x, distance=di, sli=silhouette_avg, maxi=m)


# Function to find the centers of the clusters
# REF: https://mubaris.com/2017/10/01/kmeans-clustering-in-python/
@application.route('/cluster1', methods=['GET', 'POST'])
def cluster1():
    '''con = sql.connect("database.db")
    con.row_factory = sql.Row
    cursor = con.cursor()
    cursor.execute("select cabin, fare from titanic")
    rows = cursor.fetchall()'''
    x = request.form['start']
    x = int(x)
    e = request.form['in1']
    r = request.form['in2']
    data1 = df[e]
    data2 = df[r]
    y = list(zip(data1, data2))
    '''d = DataFrame(rows)
    #print(d)'''
    lis = numpy.array(y)
    est = KMeans(n_clusters=x)
    est.fit(y)
    e = est.cluster_centers_
    dist = numpy.linalg.norm(e)
    # print(dist)
    labels = est.labels_
    uni, counts = numpy.unique(labels, return_counts=True)
    count = dict(zip(uni, counts))
    # print(labels)
    # print(count)
    di = []
    for i in range(0, x):
        for j in range(i + 1, x):
            d = numpy.linalg.norm(e[i] - e[j])
            di.append(d)
            print(i)
            print(j)
            print(d)
    '''plt.scatter(lis[:,0], lis[:, 1], c = labels, cmap='rainbow')
    plt.scatter(e[:,0] ,e[:,1], color='black')
    plt.savefig('static/scatter1.png')
    plt.show()'''
    # f = plt.figure()
    # q = mpld3.display(f)
    m = max(di)
    clusterer = KMeans(n_clusters=x, random_state=10)
    cluster_labels = clusterer.fit_predict(y)
    silhouette_avg = silhouette_score(y, cluster_labels)
    # print("For n_clusters =", x, "The average silhouette_score is :", silhouette_avg)
    return render_template('cluster1.html', rows=e, dist=dist, count=count, num=x, distance=di, sli=silhouette_avg, maxi=m)

#Function to display piechart for female survivors
@application.route('/titapie', methods=['GET', 'POST'])
def titapie():
    con = sql.connect("database.db")
    con.row_factory = sql.Row
    cursor = con.cursor()
    cursor.execute("select * from titanic WHERE survived = 1 AND sex = 'female'")
    rows = cursor.fetchall()
    a = len(rows)
    cursor.execute("select * from titanic WHERE survived = 0 AND sex = 'female'")
    rows = cursor.fetchall()
    b = len(rows)
    print(a)
    print(b)
    counts=[a,b]
    return render_template('titapie.html', var=counts)

#Function to dispaly a barchart for male survivors
@application.route('/titabar', methods=['GET', 'POST'])
def titabar():
    con = sql.connect("database.db")
    con.row_factory = sql.Row
    cursor = con.cursor()
    cursor.execute("select * from titanic WHERE survived = 1 AND sex='male'")
    rows = cursor.fetchall()
    a = len(rows)
    cursor.execute("select * from titanic WHERE survived = 0 AND sex='male'")
    rows = cursor.fetchall()
    b = len(rows)
    counts=[a,b]
    return render_template('titabar.html', var=counts)

@application.route('/imagelast', methods=['GET', 'POST'])
def imagelast():
    con = sql.connect("database.db")
    con.row_factory = sql.Row
    cursor = con.cursor()
    cursor.execute("select PictureCap from minnow WHERE Lname = :ln ", {"ln":request.form['start']})
    rows = cursor.fetchall()
    print(rows)
    return render_template('imageresult.html',rows=rows)

@application.route('/ques7', methods=['GET', 'POST'])
def ques7():
    con = sql.connect("database.db")
    con.row_factory = sql.Row
    cursor = con.cursor()
    a = request.form['start']
    a = int(a)
    b = request.form['end']
    b = int(b)
    c = request.form['start1']
    c = int(c)
    d = request.form['end1']
    d = int(d)
    cursor.execute("select cabin, fare from titanic")
    rows = cursor.fetchall()
    data1 = df['age']
    data2 = df['fare']
    y = list(zip(data1, data2))
    d = DataFrame(rows)
    #print(d)
    lis = numpy.array(y)
    est.fit(d)
    e = est.cluster_centers_
    dist = numpy.linalg.norm(e)
    # print(dist)
    labels = est.labels_
    uni, counts = numpy.unique(labels, return_counts=True)
    print(uni)
    print(counts)
    count = dict(zip(uni, counts))
    di = []
    for i in range(0, x):
        for j in range(i + 1, x):
            d = numpy.linalg.norm(e[i] - e[j])
            di.append(d)
            print(i)
            print(j)
            print(d)
    plt.scatter(lis[:,0], lis[:, 1], c = labels, cmap='rainbow')
    plt.scatter(e[:,0] ,e[:,1], color='black')
    plt.savefig('static/scatter1.png')
    plt.show()
    # f = plt.figure()
    # q = mpld3.display(f)
    m = max(di)
    clusterer = KMeans(n_clusters=x, random_state=10)
    cluster_labels = clusterer.fit_predict(y)
    silhouette_avg = silhouette_score(y, cluster_labels)
    return render_template('cluster.html', rows=e, dist=dist, count=count, num=x, distance=di, sli=silhouette_avg, maxi=m)


@application.route('/qeight', methods=['GET', 'POST'])
def qeight():
    con = sql.connect("database.db")
    con.row_factory = sql.Row
    cursor = con.cursor()
    cursor.execute("select * from minnow WHERE Survived='Y' AND Fare=100")
    rows = cursor.fetchall()
    a = len(rows)
    cursor.execute("select * from minnow WHERE Survived='N' AND Fare=100")
    rows = cursor.fetchall()
    b = len(rows)
    cursor.execute("select * from minnow WHERE Survived='Y' AND Fare=200")
    rows = cursor.fetchall()
    c = len(rows)
    cursor.execute("select * from minnow WHERE Survived='N' AND Fare=200")
    rows = cursor.fetchall()
    d = len(rows)
    cursor.execute("select * from minnow WHERE Survived='Y' AND Fare=500")
    rows = cursor.fetchall()
    e = len(rows)
    cursor.execute("select * from minnow WHERE Survived='N' AND Fare=500")
    rows = cursor.fetchall()
    f = len(rows)
    cursor.execute("select * from minnow WHERE Survived='Y' AND Fare=800")
    rows = cursor.fetchall()
    g = len(rows)
    cursor.execute("select * from minnow WHERE Survived='N' AND Fare=800")
    rows = cursor.fetchall()
    h = len(rows)
    cursor.execute("select * from minnow WHERE Survived='Y' AND Fare=2500")
    rows = cursor.fetchall()
    i = len(rows)
    print(i)
    cursor.execute("select * from minnow WHERE Survived='N' AND Fare=2500")
    rows = cursor.fetchall()
    j = len(rows)

    counts=[a,b,c,d,e,f,g,h,i,j]
    return render_template('qeightresult.html', var=counts)


if __name__ == '__main__':
    application.run(host='127.0.0.1', port=port, debug=True)
