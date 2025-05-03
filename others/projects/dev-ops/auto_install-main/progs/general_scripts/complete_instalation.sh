# complete_instalation.sh

complete_instalation() {
  program_name="$1"
  
  # Проверка наличия программы в системном пути
  if command -v "$program_name" &>/dev/null; then
    echo "$program_name установлена и настроена правильно."
    return 1
  else
    echo "При установке $program_name возникла ошибка.
        Либо название программы в конфиге указанно не верно,
        проверьте установилась ли программа. Спасибо и удачи :)"
    return 1
  fi
  
  
  echo "Не забудь поставить звезду на Github: $github_link"
  echo "Спасибо, что пользуетесь, $author_github (мне) приятно.
    Если вы нашли баг, обязательно напишите в гитхаб"
  return 0
}
