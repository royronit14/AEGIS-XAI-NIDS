from bson import ObjectId

def safe_serialize(obj):
    if isinstance(obj, list):
        return [safe_serialize(i) for i in obj]

    if isinstance(obj, dict):
        return {
            k: safe_serialize(v)
            for k, v in obj.items()
            if k != "_id"
        }

    if isinstance(obj, ObjectId):
        return str(obj)

    return obj
