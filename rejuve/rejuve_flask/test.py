from query_generator import NaturalToAnnotation

request_1 = {
  "requests": {
    "nodes": [
      { "node_id": "n1", "id": "", "type": "transript", "properties": {} }
    ],
    "predicates": []
  }
}

request_2 = {
  "requests": {
    "nodes": [
      {
        "node_id": "n1",
        "id": "",
        "type": "transcri",
        "properties": {}
      },
      { 
        "node_id": "n2", 
        "id": "", 
        "type": "prtein", 
        "properties": {} 
      }
    ],
    "predicates": [{ "type": "translates to", "source": "n1", "target": "n2" }]
  }
}

nla = NaturalToAnnotation()
res_1 = nla.request_optimizer(request_1)
res_2 = nla.request_optimizer(request_2)

print(res_1)
print(res_2)
