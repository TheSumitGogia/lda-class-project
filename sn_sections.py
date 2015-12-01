from bs4 import BeautifulSoup
from subprocess import call
from django.utils.encoding import smart_str

link_file = open('blinks.txt', 'r')
book_links = link_file.readlines()
book_links = [link.rstrip() for link in book_links]

for link in book_links:
    lsplit = link.split('/')
    title = lsplit[4]
    call(["mkdir", "books/" + title])
    call(["wget", '-O', "books/" + title + '/' + title, link])
    book_file = open("books/" + title + '/' + title, 'r')
    book_soup = BeautifulSoup(book_file)
    sections = book_soup.select('.entry a')
    sections = [blink['href'] for blink in sections if blink['href'].startswith('section')]
    for section in sections:
        sec_name = section.split('.')[0]
        sec_file = open('books/' + title + '/' + sec_name + '.txt', 'w')
        call(["wget", "-O", "section_file.html", link + section])
        sec_soup = BeautifulSoup(open('section_file.html', 'r'))
        sec_texts = sec_soup.select('.studyGuideText > p')
        sec_texts = [smart_str(sec_text.get_text()) for sec_text in sec_texts]
        sec_file.write('\n\n'.join(sec_texts))
        sec_file.close()
    book_file.close()
