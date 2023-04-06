#A
def unique(lst):
    return len(set(lst))
print(unique([1,2,3,2,1]))

#B
def same_nums(lst1, lst2):
    return len(set(lst1).intersection(set(lst2)))
print(same_nums([1,2,4], [1,2,3]))

#C
def count_numbers(lst1, lst2):
    return list(set(lst1).intersection(set(lst2)))
print(count_numbers([4,1,2,3], [1,7,2,3]))

#D
lst = [1,2,3,3,4]
seen = set()
for number in lst:
    if number in seen:
        print("YES")
    else:
        print("NO")
        seen.add(number)

#E
with open('Е.txt', 'r') as f:
    lines = f.readlines()
num_balls = list(map(int, lines[0].split()))
irina_balls = set(map(int, lines[1:num_balls[0]+1]))
igor_balls = set(map(int, lines[num_balls[0]+1:]))
common_balls = sorted(irina_balls & igor_balls)
irina_only_balls = sorted(irina_balls - igor_balls)
igor_only_balls = sorted(igor_balls - irina_balls)
print(*common_balls)
print(len(common_balls))
print(*irina_only_balls)
print(len(irina_only_balls)-1)
print(*igor_only_balls)

#F
with open('F.txt', 'r') as text:
    text_str = str(text.readlines())
    w = text_str.split(' ')
    print(len(set(w))+1)

#G
with open('G.txt', 'r') as f:
    n = int(f.readline().strip())
    nums1 = set(map(int, f.readline().strip().split()))
    answer1 = f.readline().strip()
    nums2 = set(map(int, f.readline().strip().split()))
    answer2 = f.readline().strip()
    print(' '.join(map(str, sorted(nums1.difference(nums2)))))

#I
my_dict = {}
with open('I.txt', 'r') as f:
    line = f.readline()
    j = int(line)
    while line:
        words = line.strip().split()
        for word in words:
            if not word.isdigit():
                my_dict[word] = my_dict.get(word, 0) + 1
        line = f.readline()
lng_lst = []
shr_lst = []
for key, value in my_dict.items():
    if value == j:
        lng_lst.append(key)
    if value >= 1:
        shr_lst.append(key)
h = len(lng_lst)
print(h)
for key in lng_lst:
    print(key)
k = len(shr_lst)
print(k)
for key in shr_lst:
    print(key)










    