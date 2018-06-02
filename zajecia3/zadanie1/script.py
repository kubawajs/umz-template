import emails
import classifier


def save_most_common_words(data_set, classifier):
    classifier.save_most_common_words(data_set)


emails_list = emails.Email.emails_list
save_most_common_words(emails_list, classifier.Bayes)