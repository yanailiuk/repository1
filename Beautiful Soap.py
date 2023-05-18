import requests
from bs4 import BeautifulSoup
url = "https://mof.gov.ua/uk"
try:
    r = requests.get(url)
    print(r.status_code)
    print(r.headers['content-type'])
    print(r.encoding)
    # print(r.text)
    soup = BeautifulSoup(r.text, 'lxml')
    # for child in soup.recursiveChildGenerator():
    #     if child.name:
    #         print(child.name)
    # print(soup.find_all('a', {'href': 'https://mof.gov.ua/uk'}))
    # TASKS
    print(soup.find_all('title'))
    print(soup.find_all('p'))
    print(len(soup.find_all('p')))

    first_p_tag = soup.find('p')
    text = first_p_tag.get_text()
    print(text)

    first_h2_tag = soup.find('h4')
    text1 = first_h2_tag.get_text()
    print(text1)
    print(len(text1))

    first_a_tag = soup.find('a')
    text2 = first_a_tag.get_text()
    print(text2)

    href = first_a_tag.get('href')
    print(href)

    li_tags = soup.find_all('li')
    urls = []
    for tag in li_tags:
        a_tag = tag.find('a')
        if a_tag:
            href = a_tag.get('href')
            urls.append(href)
    print(urls)

    # h2_tags = soup.find_all('h2')
    # for i in range(4):
    #     print(h2_tags[i])

    a_tags = soup.find_all('a')
    for i in range(10):
        print(a_tags[i])

#11
    tag_list = []
    for tag in soup.find_all(['h1', 'h2', 'h3']):
        tag_list.append(tag)
    print(tag_list)
# 12
    # text_elements = soup.find_all(text=True)
    # text = '\n'.join(filter(lambda x: x.strip(), text_elements))
    # print(text)
# 13
    tags = soup.find_all()
    for tag in tags:
        print(tag.name)
#14
    parent_tag = soup.find('html')
    child_tags = parent_tag.find_all()
    for tag in child_tags:
        print(tag.name)
# 15
    parent_tag = soup.find('body')
    descendants = parent_tag.descendants
    for descendant in descendants:
        if descendant.name is not None:
            print(descendant.name)
# 16
    header = soup.find("h1")
    header_text = header.text
    header_html = str(header)  # HTML-код заголовка
    header_parent_html = str(header.parent)  # HTML-код батьківського елемента заголовка

    print(header_text)
    print(header_html)
    print(header_parent_html)
# 17
    li_tags = soup.find_all("li")
    for li in li_tags:
        print(li.text)

# 23
    p_tag = soup.find("p")
    if p_tag is not None:
        p_tag.string = "Змінений текст..."
        print(p_tag.string)

# 24
    h1_tag = soup.find("h1")
    if h1_tag is not None:
        h1_tag.string = "Змінений текст..."
        new_div_tag = soup.new_tag("div")
        new_div_tag.string = "Новий текст для тегу <h1>..."
        h1_tag.append(new_div_tag)
        print(soup)

except Exception as e:
    print(f"Error {e}")


