import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="loqil1509.",
  database="crypto"
)

mycursor = mydb.cursor()
for x in range(76):
    sql = "INSERT INTO btc (num, prediction_price, price) VALUES (%s, %s, %s)"
    val = [(x, 0, 0)]
    mycursor.executemany(sql, val)
    mydb.commit()
for x in range(76):
    sql = "INSERT INTO eth (num, prediction_price, price) VALUES (%s, %s, %s)"
    val = [(x, 0, 0)]
    mycursor.executemany(sql, val)
    mydb.commit()
for x in range(76):
    sql = "INSERT INTO doge (num, prediction_price, price) VALUES (%s, %s, %s)"
    val = [(x, 0, 0)]
    mycursor.executemany(sql, val)
    mydb.commit()
for x in range(76):
    sql = "INSERT INTO ada (num, prediction_price, price) VALUES (%s, %s, %s)"
    val = [(x, 0, 0)]
    mycursor.executemany(sql, val)
    mydb.commit()
for x in range(76):
    sql = "INSERT INTO xrp (num, prediction_price, price) VALUES (%s, %s, %s)"
    val = [(x, 0, 0)]
    mycursor.executemany(sql, val)
    mydb.commit()


    