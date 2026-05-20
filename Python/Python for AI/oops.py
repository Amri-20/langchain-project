class Dog:
    def __init__(self, name, age):
        self.name=name
        self.age=age
    def bark(self, t=5):
        for i in range(t):
            print("Woof!")

class Cat:
    def __init__(self, name, color="black"):
        self.name=name
        self.color=color


jhon=Dog("jhon", 11)
jhon.name
jhon.age
jhon.bark()

marie=Cat("marie", "white")
marie.name
marie.color     #final value is used in the class


class DataValidator:
    def __init__(self):
        self.errors = []
    
    def validate_email(self, email):
        if "@" not in email:
            self.errors.append(f"Invalid email: {email}")
            return False
        return True
    
    def validate_age(self, age):
        if age < 0 or age > 150:
            self.errors.append(f"Invalid age: {age}")
            return False
        return True
    
    def get_errors(self):
        return self.errors

# Use the validator
validator = DataValidator()

# Notice: we don't pass self, just the email
validator.validate_email(email="bad-email")
validator.validate_age(age=200)

# Or using positional arguments
validator.validate_email("another-bad-email")
validator.validate_age(150)

print(validator.get_errors())
# ['Invalid email: bad-email',
#  'Invalid age: 200',
#  'Invalid email: another-bad-email']