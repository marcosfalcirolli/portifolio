import sqlite3
from flask_restful import Resource, reqparse
from flask_jwt import jwt_required

class Item(Resource):
  parser = reqparse.RequestParser()
  parser.add_argument('price',
                      type=float,
                      required=True,
                      help="This field cannot be left blank!"
                      )
#CLASSMETHODS
  @classmethod
  def find_by_name(cls, name):
    connection = sqlite3.connect('data.db')
    cursor = connection.cursor()
    query = "SELECT * FROM items WHERE name=?"
    result = cursor.execute(query, (name,))
    row = result.fetchone()
    connection.close()
    if row:
        return {'item': {'name': row[0], 'price': row[1]}}

  @classmethod
  def insert(cls, item):
      connection = sqlite3.connect('data.db')
      cursor = connection.cursor()
      query = "INSERT INTO items VALUES (?, ?)"
      cursor.execute(query, (item['name'], item['price']))
      connection.commit()
      connection.close()

  @classmethod
  def update(cls, item):
      connection = sqlite3.connect('data.db')
      cursor = connection.cursor()
      query = "UPDATE items SET price=? WHERE name=?"
      cursor.execute(query, (item['price'], item['name']))
      connection.commit()
      connection.close()

#GET
  @jwt_required()
  def get(self, name):
    item = self.find_by_name(name)
    if item:
        return item
    return {'message': 'Item not found'}, 404

#POST
  def post(self,name):
    if self.find_by_name(name):
      return {'message': "An item with name '{}' already exists.".format(name)}, 400

    data=Item.parser.parse_args()

    item = {'name':name, 'price':data['price']}
    try:
        self.insert(item)
    except:
        return {"message": "An error occured inserting the item"}, 500 #Internal Server Error

    return item, 201

#DELETE
  def delete(self, name):
      connection = sqlite3.connect('data.db')
      cursor = connection.cursor()

      query = "DELETE FROM items WHERE name=?"
      cursor.execute(query, (name,))

      connection.commit()
      connection.close()

      return {'message': 'Item deleted'}
#PUT
  def put(self, name):
      data=Item.parser.parse_args()
      item = self.find_by_name(name)
      updated_item = {'name': name, 'price': data['price']}
      if item is None:
          try:
              self.insert(updated_item)
          except:
              return {"message": "An error occured inserting the item"}, 500 #Internal Server Error
      else:
          try:
              self.update(updated_item)
          except:
              return {"message": "An error occured updating the item"}, 500 #Internal Server Error
      return updated_item

#GET
class ItemsList(Resource):
    def get(self):
        connection = sqlite3.connect('data.db')
        cursor = connection.cursor()
        query = "SELECT * FROM items"
        result = cursor.execute(query)
        items = []
        for row in result:
            items.append({'name': row[0], 'price':row[1]})
        connection.close()
        return {'items': items}
