import getopt
import sys

def main():
    # Remove the first argument from the list of command line arguments
    argument_list = sys.argv[1:]

    # Options
    options = "i:o:"

    # Long options
    long_options = ["input=", "output="]

    input_file = None
    output_file = None

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argument_list, options, long_options)
        
        # Checking each argument
        for current_argument, current_value in arguments:
            if current_argument in ("-i", "--input"):
                input_file = current_value
            elif current_argument in ("-o", "--output"):
                output_file = current_value

        if input_file and output_file:
            print(f"Input file: {input_file}")
            print(f"Output file: {output_file}")
        else:
            print("Usage: arguments_pass_demo.py -i <inputfile> -o <outputfile>")

    except getopt.error as err:
        # Output error, and return with an error code
        print(str(err))

if __name__ == "__main__":
    main()
    # nie wiem co to schodzi
    # przechodzimy proces jeszcze raz
    