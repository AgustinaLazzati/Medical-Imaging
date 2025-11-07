import zipfile

def main():

    with zipfile.ZipFile('data/Annotated.zip', "r") as z:
        z.extractall('data/Annotated')


if __name__ == "__main__":
    main()
