from flask import Flask,render_template,request
from pred import predictions
app=Flask(__name__)

@app.route('/',methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route('/input',methods=["GET","POST"])
def input():
    return render_template("input.html")

@app.route('/result',methods=["GET","POST"])
def result():
    if request.method=="POST":
        image=request.files['file']
        image_path="static/"+image.filename
        image.save(image_path)
        pred,acc=predictions.prediction(path=image_path)
        acc=max(acc[0][0],acc[0][1])
        acc=acc*100
        acc=acc.item()
        acc=round(acc,3)

    return render_template('result.html',img=image_path,n=pred,a=acc)
    
if __name__=="__main__":
    app.run()