<?php
        $data = json_decode(file_get_contents('php://input'), true);
        $content = $data["content"];
        $type = $data["type"];
        $user = $data["user_key"];
        $connection = ssh2_connect('ec2-13-209-17-85.ap-northeast-2.compute.amazonaws.com', 22);
        ssh2_auth_password($connection, 'ubuntu', 'oop712345');
	
        $sftp = ssh2_sftp($connection);
        $fb = fopen("ssh2.sftp://" . (int)$sftp . "/var/www/html/kakao/$user", 'r');
        $is_error = fgets($fb);
        fclose($fb);
	
        if($type=="photo")
        {
                if(!$sftp){
                  echo '{
                    "message":
                    {
                      "text": "fail to sftp"
                    }
                  }';
                }
                $stream = ssh2_exec($connection, 'sudo python /var/www/html/kakao/retrain_run_new.py '.$content.' '.$user);
                echo '{
                        "message":
                        {
                                "text": "어떤 음식인지 고민하고 있어요..\n 잠시만 기다려주세요."
                        },
                        "keyboard":
                        {
                                "type": "buttons",
                                "buttons": ["네"]
                        }
                }';
        }
        else if(!$is_error)
        {
                echo '
                {
                        "message":
                        {
                                "text": "사진을 먼저 보내주세요~"
                        },
                        "keyboard":
                        {
                                "type": "text"
                        }
                }';
        }
        else if($content=="네")
        {
                $fb = fopen("ssh2.sftp://" . (int)$sftp . "/var/www/html/kakao/$user", 'r');
                if(!$fb){
                  echo '{
                    "message":
                    {
                      "text": "fail to open file\n"
                    }
                  }';
                }
                $line = fgets($fb);
                $pieces = explode(" ", $line);
                $order = $pieces[0];
                $percent = $pieces[3];
                $percent = (float)$percent*100;
                fclose($fb);
                if($percent < 80)
                {
                        unlink($user);
                        echo '{
                                "message":
                                {
                                        "text": "이 음식이 무슨음식인지 잘 모르겠어요ㅠㅠ\n좀 더 잘나온 사진으로 부탁해요!!"
                                }
                        }';
                }
                else
                {
                        echo '
                        {
                                "message":
                                {
                                        "text": "제 생각에는 '.$order.'을(를) 드신것 같네요!!\n이 음식에 대해 원하는 정보를 선택해 주세요."
                                },
                                "keyboard":
                                {
                                        "type": "buttons",
                                        "buttons" : ["음식정보", "영양정보", "음식의 역사", "운동 in Postech", "끝"]
                                }
                        }';
                }
        }
        else if($content == "음식정보")
        {
		$fb = fopen("ssh2.sftp://" . (int)$sftp . "/var/www/html/kakao/$user", 'r');
                if(!$fb){
                  echo '{
                    "message":
                    {
                      "text": "fail to open file\n"
                    }
                  }';
                }
                $line = fgets($fb);
                $pieces = explode(" ", $line);
                $order = $pieces[0];
                fclose($fb);
                $foodfile=$order . "0.txt";
                $ffood=fopen("ssh2.sftp://" .(int)$sftp . "/var/www/html/kakao/food/$foodfile", 'r');
                if(!$ffood)
                {
                        echo '{
                        "message":
                        {
                                "text" : "fail to open file\n"
                        }
                        }';
                }
                $output0=fgets($ffood);
		$output0 = explode("@", $output0);
                fclose($ffood);
                echo '
                {
                        "message":
                        {
                                "text": "'.$output0[0].'"
                        },
                        "keyboard":
                        {
                                "type": "buttons",
                                "buttons" : ["음식정보", "영양정보", "음식의 역사", "운동 in Postech", "끝"]
                        }
                }';
        }
        else if($content == "영양정보")
        {
		$fb = fopen("ssh2.sftp://" . (int)$sftp . "/var/www/html/kakao/$user", 'r');
                if(!$fb){
                  echo '{
                    "message":
                    {
                      "text": "fail to open file\n"
                    }   
                  }';
                }
                $line = fgets($fb);
                $pieces = explode(" ", $line);
                $order = $pieces[0];
                fclose($fb);
                $foodfile=$order . "1.txt";
                $ffood=fopen("ssh2.sftp://" .(int)$sftp . "/var/www/html/kakao/food/$foodfile", 'r');
                if(!$ffood)
                {
                        echo '{
                        "message":
                        {
                                "text" : "fail to open file\n"
                        }
                        }';
                }
                $output1=fgets($ffood);
		$output1 = explode("@", $output1);
                fclose($ffood);
                echo '
                {
                        "message":
                        {
                                "text": "'.$output1[0].'"
                        },
                        "keyboard":
                        {
                                "type": "buttons",
                                "buttons" : ["음식정보", "영양정보", "음식의 역사", "운동 in Postech", "끝"]
                        }
                }';
        }
        else if($content == "음식의 역사")
        {
		$fb = fopen("ssh2.sftp://" . (int)$sftp . "/var/www/html/kakao/$user", 'r');
                if(!$fb){
                  echo '{
                    "message":
                    {
                      "text": "fail to open file\n"
                    }   
                  }';
                }
                $line = fgets($fb);
                $pieces = explode(" ", $line);
                $order = $pieces[0];
                fclose($fb);
                $foodfile=$order . "2.txt";
                $ffood=fopen("ssh2.sftp://" .(int)$sftp . "/var/www/html/kakao/food/$foodfile", 'r');
                if(!$ffood)
                {
                        echo '{
                                "message":
                                {
                                        "text" : "fail to open file\n"
                                }
                        }';
                }
                $output2=fgets($ffood);
                $output2 = explode("@", $output2);
                fclose($ffood);
                echo '
                {
                        "message":
                        {
                                "text": "'.$output2[0].'"
                        },
                        "keyboard":
                        {
                                "type": "buttons",
                                "buttons" : ["음식정보", "영양정보", "음식의 역사", "운동 in Postech", "끝"]
                        }
                }';
        }
        else if($content == "운동 in Postech")
        {
		$fb = fopen("ssh2.sftp://" . (int)$sftp . "/var/www/html/kakao/$user", 'r');
                if(!$fb){
                  echo '{
                    "message":
                    {
                      "text": "fail to open file\n"
                    }   
                  }';
                }
                $line = fgets($fb);
                $pieces = explode(" ", $line);
                $order = $pieces[0];
                fclose($fb);
                $foodfile=$order . "3.txt";
                $ffood=fopen("ssh2.sftp://" .(int)$sftp . "/var/www/html/kakao/food/$foodfile", 'r');
                if(!$ffood)
                {
                        echo '{
                                "message":
                                {
                                        "text" : "fail to open file\n"
                                }
                        }';
                }
                $output3=fgets($ffood);
                $output3 = explode("@", $output3);
                fclose($ffood);
                echo '
                {
                        "message":
                        {
                                "text": "'.$output3[0].'"
                        },
                        "keyboard":
                        {
                                "type": "buttons",
                                "buttons" : ["음식정보", "영양정보", "음식의 역사", "운동 in Postech", "끝"]
                        }
                }';
        }
        else if($content == "끝")
        {
                unlink($user);
                echo '{
                        "message":
                        {
                                "text": "다음에도 또 이용해주세요(씨익)"
                        },
                        "keyboard":
                        {
                                "type": "text"
                        }
                }';
        }
?>
