# -*- coding: utf-8 -*-
import math
import numpy as np
import os
from flask import Flask, render_template
from flask import request
from flask import flash
from scipy.spatial.transform import Rotation as R

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
count = 0

def mul(q1, q2):
    w = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    x = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    y = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    z = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    return [w, x, y, z]

def euler_to_quat(euler, type):
    q =[]
    i = 0
    for typeaxis in type:
        if typeaxis == 'X':
            q.append([math.cos(euler[i]/2), math.sin(euler[i]/2), 0, 0])
        elif typeaxis == 'Y':
            q.append([math.cos(euler[i]/2), 0, math.sin(euler[i]/2), 0])
        elif typeaxis == 'Z':
            q.append([math.cos(euler[i]/2), 0, 0, math.sin(euler[i]/2)])
        i += 1
    return mul(mul(q[0], q[1]),q[2])

def transform_matrix_to_euler(R, intorext, types,lock):
    rotationR = [
        [R[0, 2], R[1, 0], R[1, 1], math.pi * 0.5, R[2, 1], R[1, 1], -math.pi * 0.5, -R[1, 2], R[2, 2], R[0, 2], -R[0, 1], R[0, 0], 0],#INT_XYZ
        [R[0, 1], -R[1, 2], R[2, 2], -math.pi * 0.5, R[2, 0], R[2, 2], math.pi * 0.5, R[2, 1], R[1, 1], -R[0, 1], R[0, 2], R[0, 0], 0],#INT_XZY
        [R[1, 2], -R[0, 1], R[0, 0], -math.pi * 0.5 , R[0, 1], R[0, 0], math.pi * 0.5, R[0, 2], R[2, 2],-R[1, 2], R[1, 0], R[1, 1], 0], #INT_YXZ
        [R[1, 0], R[0, 2], R[2, 2], math.pi * 0.5, R[0, 2], R[0, 1], -math.pi * 0.5, -R[2, 0], R[0, 0], R[1, 0], -R[1, 2], R[1, 1], 0], #INT_YZX
        [R[2, 1], R[1, 0], R[0, 0], math.pi * 0.5,R[1, 0], R[0, 0], -math.pi * 0.5, -R[0, 1], R[1, 1],R[2, 1],-R[2, 0], R[2, 2], 0],#INT_ZXY
        [R[2, 0], -R[0, 1], R[1, 1], -math.pi * 0.5,R[1, 2], R[1, 1], math.pi * 0.5,R[1, 0], R[0, 0],-R[2, 0],R[2, 1], R[2, 2], 0],#INT_ZYX
        [R[0, 0], R[2, 1], R[2, 2], 0, R[1, 2], R[1, 1], math.pi, R[1, 0], -R[2, 0], R[0, 0], R[0, 1], R[0, 2], 0], #INT_XYX
        [R[0, 0], R[2, 1], R[2, 2], 0, -R[2, 1], R[2, 2], math.pi, R[2, 0], R[1, 0], R[0, 0], R[0, 2], -R[0, 1], 0],#INT_XZX
        [R[1, 1], R[0, 2], R[0, 0], 0, -R[2, 0], R[0, 0], math.pi, R[0, 1], R[2, 1], R[1, 1], R[1, 0],-R[1, 2], 0],  #INT_YXY
        [R[1, 1], R[0, 2], R[0, 0], 0, R[0, 2], -R[0, 0] ,math.pi ,R[2, 1], -R[0, 1] ,R[1, 1] ,R[1, 2], R[1, 0], 0], #INT_YZY
        [R[2, 2], R[1, 0], R[1, 1], 0, R[1, 0], R[0, 0], math.pi, R[0, 2], -R[1, 2],R[2, 2], R[2, 0], R[2, 1], 0], #INT_ZXZ
        [R[2, 2], R[1, 0], R[0, 0], 0, R[1, 0], R[0, 0], math.pi, R[1, 2], R[0, 2],R[2, 2], R[2, 1], -R[2, 0], 0], #INT_ZYZ

        [R[2, 0], -math.pi * 0.5, -R[0, 1], R[1, 1], math.pi * 0.5, R[1, 2], R[1, 1], R[2, 1], R[2, 2], -R[2, 0], R[1, 0], R[0, 0], 1], #EXT_XYZ
        [R[1, 0], math.pi * 0.5, R[0, 2], R[2, 2], -math.pi * 0.5, R[0, 2], R[0, 1], -R[1, 2], R[1, 1], R[1, 0], -R[2, 0], R[0, 0], 1], #EXT_XZY
        [R[2, 1], math.pi * 0.5, R[1, 0], R[0, 0], -math.pi * 0.5, R[1, 0], R[0, 0], -R[2, 0], R[2, 2],R[2, 1], -R[0, 1], R[1, 1], 1], #EXT_YXZcd
        [R[0, 2], -math.pi * 0.5, -R[1, 2], R[2, 2], math.pi * 0.5, R[2, 0], R[2, 2], R[0, 2], R[0, 0],-R[0, 1], R[2, 1], R[1, 1], 1], #EXT_YZX
        [R[1, 2], -math.pi * 0.5, -R[0, 1], R[0, 0], math.pi * 0.5, R[0, 1], R[0, 0], R[1, 0], R[1, 1],-R[1, 2], R[0, 2], R[2, 2], 1], #EXT_ZXY
        [R[0, 2], math.pi * 0.5, R[1, 0], R[1, 1], -math.pi * 0.5, R[2, 1], R[1, 1], -R[0, 1], R[0, 0],R[0, 2], -R[1, 2], R[2, 2], 1], #EXT_ZYX
        [R[0, 0], 0, R[2, 1], R[2, 2], math.pi, R[1, 2], R[1, 1],R[0, 1], R[0, 2], R[0, 0],R[1, 0], -R[2, 0], 1], #EXT_XYX
        [R[0, 0], 0, R[2, 1], R[2, 2], math.pi, R[2, 1], R[2, 2],R[0, 2], -R[0, 1], R[0, 0],R[2, 0], R[1, 0], 1], #EXT_XZX
        [R[1, 1], 0, R[0, 2], R[0, 0], math.pi, -R[2, 0], R[0, 0],R[1, 0], -R[1, 2], R[1, 1],R[0, 1], R[2, 1], 1], #EXT_YXY
        [R[1, 1], 0 ,R[0, 2], R[0, 0], math.pi, R[0, 2], -R[0, 0],R[1, 2], R[1, 0],R[1, 1], R[2, 1], -R[0, 1], 1], #EXT_YZY
        [R[2, 2], 0, R[1, 0], R[1, 1], math.pi, R[1, 0], R[0, 0],R[2, 0], R[2, 1], R[2, 2],R[0, 2], -R[1, 2], 1], #EXT_ZXZ
        [R[2, 2], 0, R[1, 0], R[0, 0], math.pi, R[1, 0], R[0, 0],R[2, 1], -R[2, 0], R[2, 2],R[1, 2], R[0, 2], 1] #EXT_ZYZ
    ]
    angles = [0,0,0]
    types_euler = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX", "XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ"]
    types_id = types_euler.index(types)
    if(intorext == 'INT'):
        if(abs(rotationR[types_id][0] -1) < threshold):
            lock = True
            flash('Gimbal Lock occurs. Euler angles are non-unique, we set the third angle to 0', category='message')
            angles = [math.atan2(rotationR[types_id][1],rotationR[types_id][2]), rotationR[types_id][3], 0]
            return (angles,lock)
        elif(abs(rotationR[types_id][0] + 1) < threshold):
            lock = True
            flash('Gimbal Lock occurs. Euler angles are non-unique, we set the third angle to 0')
            angles = [math.atan2(rotationR[types_id][4],rotationR[types_id][5]), rotationR[types_id][6], 0]
            return (angles,lock)
        angles[0] = math.atan2(rotationR[types_id][7],rotationR[types_id][8])
        if (types_id/6 == 1 or types_id/6 == 3):
            angles[1] = math.acos(rotationR[types_id][9])
        else:
            angles[1] = math.asin(rotationR[types_id][9])
        angles[2] = math.atan2(rotationR[types_id][10], rotationR[types_id][11])
        return (angles,lock)
    elif(intorext == 'EXT'):
        types_id +=12
        if (abs(rotationR[types_id][0] - 1) < threshold):
            lock = True
            flash("Gimbal Lock occurs. Euler angles are non-unique, we set the first angle to 0")
            angles = [0, rotationR[types_id][1],math.atan2(rotationR[types_id][2], rotationR[types_id][3])]
            return (angles,lock)
        elif (abs(rotationR[types_id][0] + 1) < threshold):
            lock = True
            flash("Gimbal Lock occurs. Euler angles are non-unique, we set the first angle to 0")
            angles = [0, rotationR[types_id][4], math.atan2(rotationR[types_id][5], rotationR[types_id][6])]
            return (angles,lock)
        angles[0] = math.atan2(rotationR[types_id][7], rotationR[types_id][8])
        if (types_id/6 == 1 or types_id/6 == 3):
            angles[1] = math.acos(rotationR[types_id][9])
        else:
           angles[1] = math.asin(rotationR[types_id][9])
        angles[2] = math.atan2(rotationR[types_id][10], rotationR[types_id][11])
    return (angles,lock)

def transform_quat_to_matrix(quat):
    q = quat.copy()
    n = np.dot(q, q)
    q = np.multiply(q, np.sqrt(2 / n))
    q = np.outer(q, q)
    rot_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0],q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0],  1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]],
    dtype = q.dtype)
    return rot_matrix



@app.route("/rotation", methods=["post"])
def rotation():
    global count
    count +=1
    if(count%10==0):
        print("count:",count)
        with open("./log.txt","w") as f:
            f.write(str(count))
    if request.form['Calibrate_method'] == "fromEuler":
        get_value[0] = [request.form['euler0'], request.form['euler1'], request.form['euler2']]
        get_value[1] = [0, 0, 0, 0]
        get_value[2] = [[0,0,0],[0,0,0],[0,0,0]]
        euler = get_value[0]
        type_s=types = request.form['euler_order']
        intorext = request.form['euler_type']
        if intorext == 'EXT':
            type_s = types[::-1]
        euler = list(map(float, euler))
        quat =  euler_to_quat(euler, type_s)
        return_value[0] = euler
        return_value[1] = quat
        matrix = transform_quat_to_matrix(quat)
        return_value[2] = matrix
        print("euler_orders=types", intorext)

        return render_template('index.html', name=get_value, name2=return_value, lock =False, euler_orders=types, euler_types=intorext, method="fromEuler")

    if request.form['Calibrate_method'] == "fromQuat":
        get_value[1] = [request.form['q0'], request.form['q1'], request.form['q2'], request.form['q3']]
        get_value[0] = [0,0,0]
        get_value[2] = [[0,0,0],[0,0,0],[0,0,0]]
        lock = False
        quat = list(map(float, get_value[1]))
        rot_matrix = transform_quat_to_matrix(quat)
        types = request.form['euler_order']
        intorext = request.form['euler_type']


        (euler,lock) = transform_matrix_to_euler(rot_matrix, intorext, types,lock)
        return_value[0] = euler
        return_value[1] = quat
        return_value[2] = rot_matrix

        return render_template('index.html', name=get_value,name2=return_value, lock =lock, euler_orders=types, euler_types=intorext, method="fromQuat")

    if request.form['Calibrate_method'] == "fromMatrix":
        get_value[2] =[ [request.form['r00'], request.form['r01'], request.form['r02']],
                        [request.form['r10'], request.form['r11'], request.form['r12']],
                        [request.form['r20'], request.form['r21'], request.form['r22']]]
        get_value[0] = [0,0,0]
        get_value[1] = [0,0,0,0]
        types = request.form['euler_order']
        intorext = request.form['euler_type']
        matrix =  R.from_matrix(get_value[2])
        lock = False
        euler = transform_matrix_to_euler(matrix, intorext, types,lock)
        quat = matrix.as_quat()
        return_value[0] = euler
        return_value[1] = quat
        return_value[2] = get_value[2]
        return render_template('index.html', name=get_value,name2=return_value, lock =lock,euler_orders=types, euler_types=intorext, method="fromMatrix")

@app.route('/')
def hello():
    global euler, quat, matrix, threshold, get_value, return_value
    euler = [0, 0, 0]
    quat = [0, 0, 0, 0]
    matrix = [[0,0,0],[0,0,0],[0,0,0]]
    threshold = 1e-6
    get_value = [euler, quat, matrix]
    return_value=[euler, quat, matrix]
    return render_template('index.html', name=get_value, name2=return_value,euler_orders="", euler_types="", method="")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 1233)


