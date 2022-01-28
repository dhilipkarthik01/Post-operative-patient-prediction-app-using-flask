from flask import Flask, request, jsonify
import pickle
import category_encoders as ce
import pandas as pd
import numpy as np

X_train = pd.read_csv('X_train.csv')
X_train = X_train.iloc[: , 1:]
col_names = list(X_train.columns.values)

encoder = ce.OrdinalEncoder(col_names)

X_train = encoder.fit_transform(X_train)
# print(X_train)

model = pickle.load(open('model.pkl','rb'))
print("model is loaded")

app = Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    a = request.args['a']
    b = request.args['b']
    c = request.args['c']
    d = request.args['d']
    e = request.args['e']
    f = request.args['f']
    g = request.args['g']
    h = request.args['h']

    print(a,b,c,d,e,f,g,h)
    df_empty = X_train[0:0]
    print(df_empty)
    # df_empty.loc[len(df.index)] = ['mid','low','good','high','stable','unstable','mod-stable',15]
    l=[a,b,c,d,e,f,g,h]
    df_empty.loc[len(df_empty.index)] = l
    # print(df_empty)
    X_test = encoder.transform(df_empty)
    # print(X_test)

    pred = model.predict(np.array(X_test).reshape(1,-1))

    strng = ""
    if pred[0] == 'A':
        strng = "Patient should be sent to general hospital floor and should be taken care"
    elif pred[0] == 'S':
        strng = "Patient is safe:) Be prepared to go home"
    elif pred[0] == 'I':
        strng = "Patient state is serious!!! Please admit to Intensive Care Unit"


    return jsonify(prediction = strng)

if __name__ == '__main__':
    app.run(debug=True)


